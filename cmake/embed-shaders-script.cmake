# read and convert a file to a hex comma-separated string
function(file_to_hex_list file_path out_hex out_len)
    file(READ "${file_path}" hex_content HEX)
    # match every two hex characters and format as 0xXX
    string(REGEX MATCHALL ".." hex_pairs "${hex_content}")
    list(TRANSFORM hex_pairs PREPEND "0x")
    string(JOIN ", " hex_list ${hex_pairs})

    # append a null terminator so it can be safely read as string if needed
    set(${out_hex} "${hex_list}, 0x00" PARENT_SCOPE)

    # track the explicit length
    list(LENGTH hex_pairs len)
    math(EXPR len "${len} + 1")
    set(${out_len} ${len} PARENT_SCOPE)
endfunction()

# process both files
file_to_hex_list("${FRAGMENT_SHADER_PATH}" FRAGMENT_SHADER_CONTENTS FRAGMENT_SHADER_LENGTH)
file_to_hex_list("${VERTEX_SHADER_PATH}" VERTEX_SHADER_CONTENTS VERTEX_SHADER_LENGTH)

# generate the final C file
configure_file("${TEMPLATE_PATH}" "${OUTPUT_PATH}" @ONLY)
