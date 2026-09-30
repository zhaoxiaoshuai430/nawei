# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

if(CMAKE_HOST_WIN32 AND CMAKE_GENERATOR MATCHES "Ninja")
	set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON)
endif()

# Add this line to disable the specific warning
add_compile_options(-Wno-missing-field-initializers)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
# "Trim" the build. Include the minimal set of components, main, and anything it depends on.
idf_build_set_property(MINIMAL_BUILD ON)

set(PROJECT_VER "2.5.0")
project(xiaozhi)

if(CMAKE_HOST_WIN32)
	foreach(component_name espressif__esp_audio_codec espressif__esp_image_effects)
		if(TARGET __idf_${component_name})
			foreach(link_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
				get_target_property(component_links __idf_${component_name} ${link_property})
				if(component_links)
					list(FILTER component_links EXCLUDE REGEX "^-L |^\\$<LINK_ONLY:-L ")
					set_property(TARGET __idf_${component_name} PROPERTY ${link_property} "${component_links}")
				endif()
			endforeach()
		endif()
	endforeach()
endif()
