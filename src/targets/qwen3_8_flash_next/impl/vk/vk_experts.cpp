#include "targets/qwen3_8_flash_next/impl/vk/vk_experts.h"

#include "targets/qwen3_8_flash_next/impl/expert_bank.h"

#include "down_spv.h"
#include "gate_up_spv.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr std::int32_t kHidden        = 2560;
constexpr std::int32_t kIntermediate  = 640;
constexpr std::uint32_t kMaxJobs      = 10;  // == kTopK in expert_cache.cpp
constexpr std::uint32_t kMaxTokens    = 8;   // == FlashNextOffloadPolicy::host_max_tokens ceiling
constexpr std::uint32_t kPromoBuffers = 3;
constexpr std::size_t kOneGiB         = 1ULL << 30; // slot pool cap: one VkBuffer/allocation

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') { return fallback; }
    return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
}

std::size_t align256(std::size_t v) { return (v + 255U) & ~std::size_t{255U}; }

#define VK_CHECK(expr)                                                                                               \
    do {                                                                                                             \
        const VkResult vk_check_result_ = (expr);                                                                    \
        if (vk_check_result_ != VK_SUCCESS) {                                                                        \
            throw std::runtime_error(std::string("Flash-Next Vulkan expert tier: ") + #expr +                       \
                                     " failed with VkResult " + std::to_string(static_cast<int>(vk_check_result_))); \
        }                                                                                                             \
    } while (false)

// One packed expert's byte layout inside the slot pool, matching
// FlashNextExpertCache::packed_bytes()/pack() exactly (both are the same formula over the same
// bank geometry, kept independently: see AGENTS.md on closed per-boundary math implementations).
struct SlotLayout {
    std::uint32_t gate_code_words  = 0;
    std::uint32_t gate_scale_words = 0;
    std::uint32_t down_code_words  = 0;
    std::uint32_t down_scale_words = 0;
    std::size_t packed_bytes       = 0;

    static SlotLayout from(const MoeWeights& weights) {
        SlotLayout out;
        out.gate_code_words  = static_cast<std::uint32_t>(weights.expert_gate_up.code_bytes_per_expert / 4);
        out.gate_scale_words = static_cast<std::uint32_t>(weights.expert_gate_up.scale_bytes_per_expert / 4);
        out.down_code_words  = static_cast<std::uint32_t>(weights.expert_down.code_bytes_per_expert / 4);
        out.down_scale_words = static_cast<std::uint32_t>(weights.expert_down.scale_bytes_per_expert / 4);
        out.packed_bytes     = align256(weights.expert_gate_up.code_bytes_per_expert) +
                            align256(weights.expert_gate_up.scale_bytes_per_expert) + 256 +
                            align256(weights.expert_down.code_bytes_per_expert) +
                            align256(weights.expert_down.scale_bytes_per_expert) + 256;
        return out;
    }
};

struct Buffer {
    VkBuffer buffer             = VK_NULL_HANDLE;
    VkDeviceMemory memory       = VK_NULL_HANDLE;
    void* mapped                = nullptr;
    std::size_t bytes           = 0;
};

} // namespace

struct FlashNextVkExperts::Impl {
    VkInstance instance             = VK_NULL_HANDLE;
    VkPhysicalDevice physical        = VK_NULL_HANDLE;
    VkDevice device                  = VK_NULL_HANDLE;
    VkQueue queue                    = VK_NULL_HANDLE;
    std::uint32_t queue_family        = 0;
    VkCommandPool cmd_pool            = VK_NULL_HANDLE;
    VkCommandBuffer compute_cmd       = VK_NULL_HANDLE;
    VkFence compute_fence             = VK_NULL_HANDLE;

    VkDescriptorSetLayout set_layout  = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout  = VK_NULL_HANDLE;
    VkShaderModule gate_up_module     = VK_NULL_HANDLE;
    VkShaderModule down_module        = VK_NULL_HANDLE;
    VkPipeline gate_up_pipeline       = VK_NULL_HANDLE;
    VkPipeline down_pipeline          = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool        = VK_NULL_HANDLE;
    VkDescriptorSet desc_set          = VK_NULL_HANDLE;

    Buffer slot_pool, x_buf, jobs_buf, act_buf, out_buf;
    // Device-local copies the shaders read (the host-visible originals live in system RAM, which
    // the RX 570 would otherwise read across PCIe for every element).
    Buffer x_dev, jobs_dev;
    std::array<Buffer, kPromoBuffers> staging{};
    std::array<VkCommandBuffer, kPromoBuffers> promo_cmd{};
    std::array<VkFence, kPromoBuffers> promo_fence{};
    std::array<bool, kPromoBuffers> promo_busy{};

    SlotLayout layout;

    struct Slot {
        std::int64_t key        = -1;
        std::uint64_t last_used = 0;
        std::uint32_t frequency = 0; // hits, halved every kDecayCalls submits
        bool pending             = false;
    };
    struct Promotion {
        std::uint32_t slot = 0;
        std::int64_t key   = 0;
        std::size_t buffer = 0;
    };
    std::vector<Slot> slot;
    std::unordered_map<std::int64_t, std::uint32_t> resident;
    std::vector<Promotion> promotions;
    std::uint64_t clock = 0;
    std::uint64_t submits = 0;

    bool pending_submit = false;

    ~Impl() {
        if (device == VK_NULL_HANDLE) { return; }
        (void)vkDeviceWaitIdle(device);
        const auto destroy_buffer = [&](Buffer& b) {
            if (b.mapped != nullptr) { vkUnmapMemory(device, b.memory); }
            if (b.buffer != VK_NULL_HANDLE) { vkDestroyBuffer(device, b.buffer, nullptr); }
            if (b.memory != VK_NULL_HANDLE) { vkFreeMemory(device, b.memory, nullptr); }
        };
        destroy_buffer(slot_pool);
        destroy_buffer(x_buf);
        destroy_buffer(x_dev);
        destroy_buffer(jobs_dev);
        destroy_buffer(jobs_buf);
        destroy_buffer(act_buf);
        destroy_buffer(out_buf);
        for (auto& b : staging) { destroy_buffer(b); }
        for (auto f : promo_fence) {
            if (f != VK_NULL_HANDLE) { vkDestroyFence(device, f, nullptr); }
        }
        if (compute_fence != VK_NULL_HANDLE) { vkDestroyFence(device, compute_fence, nullptr); }
        if (desc_pool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device, desc_pool, nullptr); }
        if (gate_up_pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, gate_up_pipeline, nullptr); }
        if (down_pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, down_pipeline, nullptr); }
        if (pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, pipeline_layout, nullptr); }
        if (set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, set_layout, nullptr); }
        if (gate_up_module != VK_NULL_HANDLE) { vkDestroyShaderModule(device, gate_up_module, nullptr); }
        if (down_module != VK_NULL_HANDLE) { vkDestroyShaderModule(device, down_module, nullptr); }
        if (cmd_pool != VK_NULL_HANDLE) { vkDestroyCommandPool(device, cmd_pool, nullptr); }
        vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) { vkDestroyInstance(instance, nullptr); }
    }
};

namespace {

// Flat job table entry (gate_up.comp/down.comp read this same layout as a plain uint[] SSBO, not
// a GLSL struct array, so there is no std430 struct/array padding rule to keep in sync by hand):
// slot, count, tokens[kMaxTokens], weights[kMaxTokens] (bit-cast float), all 4-byte words, no
// padding.
struct GpuJob {
    std::uint32_t slot;
    std::uint32_t count;
    std::uint32_t tokens[kMaxTokens];
    float weights[kMaxTokens];
};
static_assert(sizeof(GpuJob) == (2 + 2 * kMaxTokens) * sizeof(std::uint32_t),
             "GpuJob must be exactly 2 + 2*kMaxTokens words with no padding");

std::uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties& props, std::uint32_t type_bits,
                               VkMemoryPropertyFlags required) {
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) == 0) { continue; }
        if ((props.memoryTypes[i].propertyFlags & required) == required) { return i; }
    }
    throw std::runtime_error("Flash-Next Vulkan expert tier: no matching memory type");
}

Buffer make_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties& mem_props, std::size_t bytes,
                   VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool map) {
    Buffer out;
    out.bytes = bytes;
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size        = bytes;
    buffer_info.usage       = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &buffer_info, nullptr, &out.buffer));
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize  = requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_props, requirements.memoryTypeBits, properties);
    VK_CHECK(vkAllocateMemory(device, &alloc_info, nullptr, &out.memory));
    VK_CHECK(vkBindBufferMemory(device, out.buffer, out.memory, 0));
    if (map) { VK_CHECK(vkMapMemory(device, out.memory, 0, bytes, 0, &out.mapped)); }
    return out;
}

VkShaderModule make_shader(VkDevice device, const unsigned char* bytes, std::size_t size) {
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode    = reinterpret_cast<const std::uint32_t*>(bytes);
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

VkPipeline make_pipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule module,
                        const SlotLayout& slot_layout) {
    const std::array<std::uint32_t, 4> values{slot_layout.gate_code_words, slot_layout.gate_scale_words,
                                              slot_layout.down_code_words, slot_layout.down_scale_words};
    const std::array<VkSpecializationMapEntry, 4> entries{
        VkSpecializationMapEntry{0, 0 * sizeof(std::uint32_t), sizeof(std::uint32_t)},
        VkSpecializationMapEntry{1, 1 * sizeof(std::uint32_t), sizeof(std::uint32_t)},
        VkSpecializationMapEntry{2, 2 * sizeof(std::uint32_t), sizeof(std::uint32_t)},
        VkSpecializationMapEntry{3, 3 * sizeof(std::uint32_t), sizeof(std::uint32_t)},
    };
    VkSpecializationInfo spec{};
    spec.mapEntryCount = static_cast<std::uint32_t>(entries.size());
    spec.pMapEntries   = entries.data();
    spec.dataSize      = values.size() * sizeof(std::uint32_t);
    spec.pData         = values.data();

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module               = module;
    stage.pName                = "main";
    stage.pSpecializationInfo = &spec;

    VkComputePipelineCreateInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage  = stage;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    return pipeline;
}

} // namespace

FlashNextVkPolicy FlashNextVkPolicy::from_environment() {
    FlashNextVkPolicy out;
    out.enabled          = env_u32("NINFER_FLASH_NEXT_VK", 1) != 0;
    out.device_index     = static_cast<std::int32_t>(env_u32("NINFER_FLASH_NEXT_VK_DEVICE", 0xFFFFFFFFU));
    out.cache_bytes       = static_cast<std::size_t>(env_u32("NINFER_FLASH_NEXT_VK_CACHE_MB", 0)) << 20;
    out.reserve_mib       = env_u32("NINFER_FLASH_NEXT_VK_RESERVE_MB", static_cast<std::uint32_t>(out.reserve_mib));
    out.promote_per_call = env_u32("NINFER_FLASH_NEXT_VK_PROMOTE", out.promote_per_call);
    out.admit_misses      = std::max(1U, env_u32("NINFER_FLASH_NEXT_VK_ADMIT", out.admit_misses));
    out.pin               = env_u32("NINFER_FLASH_NEXT_VK_PIN", out.pin ? 1U : 0U) != 0;
    if (const char* v = std::getenv("NINFER_FLASH_NEXT_VK_PIN_PROFILE"); v != nullptr && v[0] != '\0') {
        out.pin_profile = v;
    }
    out.report            = env_u32("NINFER_FLASH_NEXT_OFFLOAD_STATS", 0) != 0;
    return out;
}

FlashNextVkExperts::FlashNextVkExperts(std::unique_ptr<Impl> impl, FlashNextVkPolicy policy, std::size_t packed_bytes,
                                       std::uint32_t slots)
    : impl_(std::move(impl)), policy_(policy), packed_bytes_(packed_bytes), slots_(slots) {}

FlashNextVkExperts::~FlashNextVkExperts() = default;

std::unique_ptr<FlashNextVkExperts> FlashNextVkExperts::create(const MoeWeights& weights, FlashNextVkPolicy policy) {
    if (!policy.enabled) { return nullptr; }
    try {
        auto impl = std::make_unique<Impl>();

        VkApplicationInfo app_info{};
        app_info.sType         = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "ninfer-flash-next";
        app_info.apiVersion    = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instance_info{};
        instance_info.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;
        if (vkCreateInstance(&instance_info, nullptr, &impl->instance) != VK_SUCCESS) {
            std::fprintf(stderr, "[flash-next vk] no usable Vulkan loader/ICD; running without the RX 570 tier\n");
            return nullptr;
        }

        std::uint32_t device_count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(impl->instance, &device_count, nullptr));
        std::vector<VkPhysicalDevice> devices(device_count);
        VK_CHECK(vkEnumeratePhysicalDevices(impl->instance, &device_count, devices.data()));
        std::vector<VkPhysicalDevice> amd_devices;
        for (auto d : devices) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.vendorID == 0x1002) { amd_devices.push_back(d); }
        }
        if (amd_devices.empty()) {
            std::fprintf(stderr, "[flash-next vk] no AMD Vulkan device found; running without the RX 570 tier\n");
            return nullptr;
        }
        const std::size_t index = (policy.device_index < 0 ||
                                   static_cast<std::size_t>(policy.device_index) >= amd_devices.size())
                                     ? 0
                                     : static_cast<std::size_t>(policy.device_index);
        impl->physical           = amd_devices[index];

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(impl->physical, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(impl->physical, &family_count, families.data());
        std::int32_t compute_only = -1, any_compute = -1;
        for (std::uint32_t i = 0; i < family_count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) { continue; }
            if (any_compute < 0) { any_compute = static_cast<std::int32_t>(i); }
            if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 && compute_only < 0) {
                compute_only = static_cast<std::int32_t>(i);
            }
        }
        if (any_compute < 0) {
            std::fprintf(stderr, "[flash-next vk] AMD device has no compute queue; running without the RX 570 tier\n");
            return nullptr;
        }
        impl->queue_family = static_cast<std::uint32_t>(compute_only >= 0 ? compute_only : any_compute);

        const float priority = 1.0F;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = impl->queue_family;
        queue_info.queueCount        = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{};
        device_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount     = 1;
        device_info.pQueueCreateInfos        = &queue_info;
        VK_CHECK(vkCreateDevice(impl->physical, &device_info, nullptr, &impl->device));
        vkGetDeviceQueue(impl->device, impl->queue_family, 0, &impl->queue);

        VkPhysicalDeviceMemoryProperties mem_props{};
        vkGetPhysicalDeviceMemoryProperties(impl->physical, &mem_props);

        // Budget: VK_EXT_memory_budget on the DEVICE_LOCAL heap, less reserve_mib (the desktop
        // compositor also lives on this GPU); NINFER_FLASH_NEXT_VK_CACHE_MB overrides outright.
        std::size_t budget_bytes = policy.cache_bytes;
        if (budget_bytes == 0) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props{};
            budget_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
            VkPhysicalDeviceMemoryProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            props2.pNext = &budget_props;
            vkGetPhysicalDeviceMemoryProperties2(impl->physical, &props2);
            std::uint32_t device_local_heap = 0;
            for (std::uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
                if ((mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                    device_local_heap = i;
                    break;
                }
            }
            const std::size_t reserve = (policy.reserve_mib << 20);
            const std::size_t heap_budget = budget_props.heapBudget[device_local_heap];
            budget_bytes                   = heap_budget > reserve ? heap_budget - reserve : 0;
        }
        budget_bytes = std::min(budget_bytes, kOneGiB);

        impl->layout          = SlotLayout::from(weights);
        const std::uint32_t slots = static_cast<std::uint32_t>(budget_bytes / impl->layout.packed_bytes);
        if (slots == 0) {
            std::fprintf(stderr,
                         "[flash-next vk] %zu MiB budget holds no experts (%zu bytes each); running without the "
                         "RX 570 tier\n",
                         budget_bytes >> 20, impl->layout.packed_bytes);
            return nullptr;
        }

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = impl->queue_family;
        VK_CHECK(vkCreateCommandPool(impl->device, &pool_info, nullptr, &impl->cmd_pool));

        const auto alloc_cmd = [&](VkCommandBuffer& out_cmd) {
            VkCommandBufferAllocateInfo info{};
            info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            info.commandPool        = impl->cmd_pool;
            info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            info.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(impl->device, &info, &out_cmd));
        };
        alloc_cmd(impl->compute_cmd);
        // Created signaled: submit()/begin_promotion() wait on these before their first use, and
        // no command buffer has been submitted yet.
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(impl->device, &fence_info, nullptr, &impl->compute_fence));
        for (std::uint32_t i = 0; i < kPromoBuffers; ++i) {
            alloc_cmd(impl->promo_cmd[i]);
            VK_CHECK(vkCreateFence(impl->device, &fence_info, nullptr, &impl->promo_fence[i]));
            impl->staging[i] = make_buffer(impl->device, mem_props, impl->layout.packed_bytes,
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                           true);
        }

        const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const VkMemoryPropertyFlags device_local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        impl->slot_pool = make_buffer(impl->device, mem_props, static_cast<std::size_t>(slots) * impl->layout.packed_bytes,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     device_local, false);
        impl->x_buf = make_buffer(impl->device, mem_props, std::size_t{kMaxTokens} * kHidden * sizeof(float),
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, true);
        impl->jobs_buf = make_buffer(impl->device, mem_props, std::size_t{kMaxJobs} * sizeof(GpuJob),
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, true);
        impl->x_dev = make_buffer(impl->device, mem_props, impl->x_buf.bytes,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, device_local,
                                  false);
        impl->jobs_dev = make_buffer(impl->device, mem_props, impl->jobs_buf.bytes,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     device_local, false);
        impl->act_buf = make_buffer(impl->device, mem_props,
                                    std::size_t{kMaxJobs} * kMaxTokens * kIntermediate * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, device_local, false);
        impl->out_buf = make_buffer(impl->device, mem_props, std::size_t{kMaxTokens} * kHidden * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, true);

        std::memset(impl->jobs_buf.mapped, 0, impl->jobs_buf.bytes);

        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding         = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_layout_info{};
        set_layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set_layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        set_layout_info.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(impl->device, &set_layout_info, nullptr, &impl->set_layout));

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts    = &impl->set_layout;
        VK_CHECK(vkCreatePipelineLayout(impl->device, &pipeline_layout_info, nullptr, &impl->pipeline_layout));

        impl->gate_up_module = make_shader(impl->device, gate_up_bytes, gate_up_size);
        impl->down_module    = make_shader(impl->device, down_bytes, down_size);
        impl->gate_up_pipeline = make_pipeline(impl->device, impl->pipeline_layout, impl->gate_up_module, impl->layout);
        impl->down_pipeline    = make_pipeline(impl->device, impl->pipeline_layout, impl->down_module, impl->layout);

        VkDescriptorPoolSize pool_size{};
        pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = static_cast<std::uint32_t>(bindings.size());
        VkDescriptorPoolCreateInfo desc_pool_info{};
        desc_pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        desc_pool_info.maxSets       = 1;
        desc_pool_info.poolSizeCount = 1;
        desc_pool_info.pPoolSizes    = &pool_size;
        VK_CHECK(vkCreateDescriptorPool(impl->device, &desc_pool_info, nullptr, &impl->desc_pool));
        VkDescriptorSetAllocateInfo desc_alloc_info{};
        desc_alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc_info.descriptorPool     = impl->desc_pool;
        desc_alloc_info.descriptorSetCount = 1;
        desc_alloc_info.pSetLayouts        = &impl->set_layout;
        VK_CHECK(vkAllocateDescriptorSets(impl->device, &desc_alloc_info, &impl->desc_set));

        const std::array<Buffer*, 5> set_buffers{&impl->slot_pool, &impl->x_dev, &impl->jobs_dev, &impl->act_buf,
                                                 &impl->out_buf};
        std::array<VkDescriptorBufferInfo, 5> buffer_infos{};
        std::array<VkWriteDescriptorSet, 5> writes{};
        for (std::uint32_t i = 0; i < set_buffers.size(); ++i) {
            buffer_infos[i] = {set_buffers[i]->buffer, 0, VK_WHOLE_SIZE};
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = impl->desc_set;
            writes[i].dstBinding      = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &buffer_infos[i];
        }
        vkUpdateDescriptorSets(impl->device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

        impl->slot.assign(slots, Impl::Slot{});
        const std::size_t packed_bytes = impl->layout.packed_bytes;
        std::fprintf(stderr,
                     "[flash-next vk] RX 570 expert tier: %u experts (%.2f GiB) on queue family %u\n", slots,
                     static_cast<double>(slots) * packed_bytes / static_cast<double>(1U << 30), impl->queue_family);
        return std::unique_ptr<FlashNextVkExperts>(new FlashNextVkExperts(std::move(impl), policy, packed_bytes, slots));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[flash-next vk] init failed (%s); running without the RX 570 tier\n", error.what());
        return nullptr;
    }
}

bool FlashNextVkExperts::resident(std::int64_t key) const { return impl_->resident.contains(key); }

std::uint32_t FlashNextVkExperts::slot_of(std::int64_t key) const { return impl_->resident.at(key); }

void FlashNextVkExperts::report() const {
    if (stats_.calls == 0) { return; }
    std::fprintf(stderr, "[flash-next vk] calls %llu hits %llu promotions %llu submit %.3fs\n",
                 static_cast<unsigned long long>(stats_.calls), static_cast<unsigned long long>(stats_.hits),
                 static_cast<unsigned long long>(stats_.promotions), stats_.submit_seconds);
}

void FlashNextVkExperts::submit(std::span<const FlashNextVkJob> jobs, const std::uint16_t* x_bf16,
                                std::int32_t tokens) {
    if (jobs.empty()) { return; }
    ++stats_.calls;
    stats_.hits += jobs.size();
    auto& impl = *impl_;

    // BF16 -> FP32 (the device has no FP16 arithmetic); x is [tokens][kHidden].
    auto* xf = static_cast<float*>(impl.x_buf.mapped);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < kHidden; ++c) {
            const std::uint32_t bits = static_cast<std::uint32_t>(x_bf16[t * kHidden + c]) << 16;
            float value               = 0.0F;
            std::memcpy(&value, &bits, sizeof(value));
            xf[t * kHidden + c] = value;
        }
    }

    // Hit frequencies for the admission test; halved every kDecayCalls submits (~a few dozen
    // decode tokens of 48 layers) so the tier follows the routing.
    constexpr std::uint64_t kDecayCalls = 2048;
    ++impl.clock;
    for (const auto& job : jobs) {
        auto& slot     = impl.slot[job.slot];
        slot.frequency = std::min<std::uint32_t>(slot.frequency + 1, 1U << 20);
        slot.last_used = impl.clock;
    }
    if (++impl.submits % kDecayCalls == 0) {
        for (auto& slot : impl.slot) { slot.frequency /= 2; }
    }

    auto* gpu_jobs = static_cast<GpuJob*>(impl.jobs_buf.mapped);
    std::memset(gpu_jobs, 0, impl.jobs_buf.bytes);
    if (jobs.size() > kMaxJobs) { throw std::logic_error("Flash-Next Vulkan expert tier: too many jobs in one call"); }
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        const auto& job = jobs[j];
        if (job.columns.size() != job.weights.size() || job.columns.empty() || job.columns.size() > kMaxTokens) {
            throw std::invalid_argument("Flash-Next Vulkan expert tier: invalid job");
        }
        gpu_jobs[j].slot  = job.slot;
        gpu_jobs[j].count = static_cast<std::uint32_t>(job.columns.size());
        for (std::size_t k = 0; k < job.columns.size(); ++k) {
            gpu_jobs[j].tokens[k]  = static_cast<std::uint32_t>(job.columns[k]);
            gpu_jobs[j].weights[k] = job.weights[k];
        }
    }

    VK_CHECK(vkWaitForFences(impl.device, 1, &impl.compute_fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(impl.device, 1, &impl.compute_fence));
    VK_CHECK(vkResetCommandBuffer(impl.compute_cmd, 0));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(impl.compute_cmd, &begin_info));
    {
        const VkBufferCopy x_copy{0, 0, static_cast<VkDeviceSize>(tokens) * kHidden * sizeof(float)};
        vkCmdCopyBuffer(impl.compute_cmd, impl.x_buf.buffer, impl.x_dev.buffer, 1, &x_copy);
        const VkBufferCopy jobs_copy{0, 0, impl.jobs_buf.bytes};
        vkCmdCopyBuffer(impl.compute_cmd, impl.jobs_buf.buffer, impl.jobs_dev.buffer, 1, &jobs_copy);
        VkMemoryBarrier copied{};
        copied.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(impl.compute_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &copied, 0, nullptr, 0, nullptr);
    }
    vkCmdBindDescriptorSets(impl.compute_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl.pipeline_layout, 0, 1,
                            &impl.desc_set, 0, nullptr);
    vkCmdBindPipeline(impl.compute_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl.gate_up_pipeline);
    vkCmdDispatch(impl.compute_cmd, kMaxJobs, kIntermediate, 1); // (job, gate/up pair)

    VkMemoryBarrier barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(impl.compute_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &barrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(impl.compute_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl.down_pipeline);
    vkCmdDispatch(impl.compute_cmd, kHidden / 4, 1, 1); // 4 output rows per workgroup
    VK_CHECK(vkEndCommandBuffer(impl.compute_cmd));

    VkSubmitInfo submit_info{};
    submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers    = &impl.compute_cmd;
    VK_CHECK(vkQueueSubmit(impl.queue, 1, &submit_info, impl.compute_fence));
    impl.pending_submit = true;
}

void FlashNextVkExperts::wait(float* accumulate_into, std::int32_t tokens) {
    if (!impl_->pending_submit) { return; }
    const auto start = std::chrono::steady_clock::now();
    // Spin briefly: the dispatch usually finishes while the CPU computes its own experts, and a
    // blocking fence wait costs a kernel wake-up (tens of microseconds) on every call.
    VkResult status = vkGetFenceStatus(impl_->device, impl_->compute_fence);
    while (status == VK_NOT_READY &&
           std::chrono::steady_clock::now() - start < std::chrono::microseconds(200)) {
        status = vkGetFenceStatus(impl_->device, impl_->compute_fence);
    }
    if (status == VK_NOT_READY) {
        VK_CHECK(vkWaitForFences(impl_->device, 1, &impl_->compute_fence, VK_TRUE, UINT64_MAX));
    } else {
        VK_CHECK(status);
    }
    impl_->pending_submit = false;
    stats_.submit_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto* out = static_cast<const float*>(impl_->out_buf.mapped);
    for (std::size_t i = 0; i < static_cast<std::size_t>(tokens) * kHidden; ++i) { accumulate_into[i] += out[i]; }
}

bool FlashNextVkExperts::begin_promotion(std::int64_t key, std::uint32_t candidate_frequency,
                                         const std::function<void(std::byte*)>& fill) {
    auto& impl = *impl_;
    if (impl.resident.contains(key)) { return false; }
    for (const auto& p : impl.promotions) {
        if (p.key == key) { return false; }
    }
    std::size_t buffer = kPromoBuffers;
    for (std::size_t i = 0; i < kPromoBuffers; ++i) {
        if (!impl.promo_busy[i]) {
            buffer = i;
            break;
        }
    }
    if (buffer == kPromoBuffers) { return false; }

    // Free slot first, else the least frequently used (ties: least recently used).
    std::int64_t victim_slot = -1;
    for (std::uint32_t s = 0; s < slots_; ++s) {
        const auto& slot = impl.slot[s];
        if (slot.pending) { continue; }
        if (slot.key < 0) {
            victim_slot = s;
            break;
        }
        if (victim_slot < 0) {
            victim_slot = s;
            continue;
        }
        const auto& best = impl.slot[static_cast<std::size_t>(victim_slot)];
        if (slot.frequency < best.frequency ||
            (slot.frequency == best.frequency && slot.last_used < best.last_used)) {
            victim_slot = s;
        }
    }
    if (victim_slot < 0) { return false; }
    {
        const auto& victim = impl.slot[static_cast<std::size_t>(victim_slot)];
        if (victim.key >= 0 && victim.frequency >= candidate_frequency) { return false; }
    }

    if (const std::int64_t old_key = impl.slot[static_cast<std::size_t>(victim_slot)].key; old_key >= 0) {
        if (const auto it = impl.resident.find(old_key);
            it != impl.resident.end() && it->second == static_cast<std::uint32_t>(victim_slot)) {
            impl.resident.erase(it);
        }
    }
    ++impl.clock;
    auto& slot     = impl.slot[static_cast<std::size_t>(victim_slot)];
    slot.key       = key;
    slot.pending   = true;
    slot.last_used = impl.clock;
    // A new entry starts at its miss count, so it is not the next victim before it is used.
    slot.frequency = candidate_frequency;

    VK_CHECK(vkWaitForFences(impl.device, 1, &impl.promo_fence[buffer], VK_TRUE, UINT64_MAX));
    fill(static_cast<std::byte*>(impl.staging[buffer].mapped));
    VK_CHECK(vkResetFences(impl.device, 1, &impl.promo_fence[buffer]));
    VK_CHECK(vkResetCommandBuffer(impl.promo_cmd[buffer], 0));
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(impl.promo_cmd[buffer], &begin_info));
    VkBufferCopy copy{};
    copy.size      = packed_bytes_;
    copy.dstOffset = static_cast<VkDeviceSize>(victim_slot) * packed_bytes_;
    vkCmdCopyBuffer(impl.promo_cmd[buffer], impl.staging[buffer].buffer, impl.slot_pool.buffer, 1, &copy);
    VK_CHECK(vkEndCommandBuffer(impl.promo_cmd[buffer]));

    VkSubmitInfo submit_info{};
    submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers    = &impl.promo_cmd[buffer];
    VK_CHECK(vkQueueSubmit(impl.queue, 1, &submit_info, impl.promo_fence[buffer]));
    impl.promo_busy[buffer] = true;
    impl.promotions.push_back({static_cast<std::uint32_t>(victim_slot), key, buffer});
    ++stats_.promotions;
    return true;
}

void FlashNextVkExperts::drop(std::int64_t key) {
    auto& impl = *impl_;
    const auto it = impl.resident.find(key);
    if (it == impl.resident.end()) { return; }
    auto& slot     = impl.slot[it->second];
    slot.key       = -1;
    slot.frequency = 0;
    impl.resident.erase(it);
}

void FlashNextVkExperts::retire_promotions() {
    auto& impl = *impl_;
    for (std::size_t i = 0; i < impl.promotions.size();) {
        const VkResult status = vkGetFenceStatus(impl.device, impl.promo_fence[impl.promotions[i].buffer]);
        if (status == VK_NOT_READY) {
            ++i;
            continue;
        }
        VK_CHECK(status);
        auto& slot   = impl.slot[impl.promotions[i].slot];
        slot.pending = false;
        impl.resident[impl.promotions[i].key]       = impl.promotions[i].slot;
        impl.promo_busy[impl.promotions[i].buffer] = false;
        impl.promotions[i]                          = impl.promotions.back();
        impl.promotions.pop_back();
    }
}

bool FlashNextVkExperts::promotion_in_flight(std::int64_t key) const {
    for (const auto& p : impl_->promotions) {
        if (p.key == key) { return true; }
    }
    return false;
}

bool FlashNextVkExperts::has_free_promotion_slot() const {
    for (bool busy : impl_->promo_busy) {
        if (!busy) { return true; }
    }
    return false;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
