# Compiles a GLSL compute shader to SPIR-V with glslangValidator and generates a header that
# embeds it as a byte array (${symbol}_bytes / ${symbol}_size), so the built binary needs neither
# the .comp/.spv files nor a shader compiler at runtime. Appends the generated header to
# ${out_headers_var} (caller adds the target's binary dir to its include path and #includes it).
function(ninfer_embed_spirv_shader out_headers_var comp_file symbol)
  find_program(GLSLANG_VALIDATOR glslangValidator)
  if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found; required to compile ${comp_file} "
                        "(NINFER_VULKAN_EXPERTS=ON needs the ninfer-sm70-vk image / libvulkan-dev "
                        "+ glslang-tools)")
  endif()
  get_filename_component(name ${comp_file} NAME_WE)
  get_filename_component(comp_file_abs ${comp_file} ABSOLUTE)
  set(spv_file ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv)
  set(header_file ${CMAKE_CURRENT_BINARY_DIR}/${name}_spv.h)
  add_custom_command(
    OUTPUT ${spv_file}
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.2 -o ${spv_file} ${comp_file_abs}
    DEPENDS ${comp_file_abs}
    COMMENT "Compiling ${comp_file} to SPIR-V")
  add_custom_command(
    OUTPUT ${header_file}
    COMMAND ${CMAKE_COMMAND}
            -DSPV_FILE=${spv_file} -DHEADER_FILE=${header_file} -DSYMBOL=${symbol}
            -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EmbedSpirvGenerate.cmake
    DEPENDS ${spv_file} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/EmbedSpirvGenerate.cmake
    COMMENT "Embedding ${name}.spv as ${symbol}")
  set(${out_headers_var} ${${out_headers_var}} ${header_file} PARENT_SCOPE)
endfunction()
