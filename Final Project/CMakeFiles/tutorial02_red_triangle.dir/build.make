# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.7/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.7/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial09_vbo_indexing"

# Include any dependencies generated for this target.
include CMakeFiles/tutorial02_red_triangle.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/tutorial02_red_triangle.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/tutorial02_red_triangle.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tutorial02_red_triangle.dir/flags.make

CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o: CMakeFiles/tutorial02_red_triangle.dir/flags.make
CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o: /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/ogl-master\ 2/tutorial02_red_triangle/tutorial02.cpp
CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o: CMakeFiles/tutorial02_red_triangle.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial09_vbo_indexing/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o -MF CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o.d -o CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o -c "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial02_red_triangle/tutorial02.cpp"

CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial02_red_triangle/tutorial02.cpp" > CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.i

CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial02_red_triangle/tutorial02.cpp" -o CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.s

CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o: CMakeFiles/tutorial02_red_triangle.dir/flags.make
CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o: /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/ogl-master\ 2/common/shader.cpp
CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o: CMakeFiles/tutorial02_red_triangle.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial09_vbo_indexing/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o -MF CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o.d -o CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o -c "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/common/shader.cpp"

CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/common/shader.cpp" > CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.i

CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/common/shader.cpp" -o CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.s

# Object files for target tutorial02_red_triangle
tutorial02_red_triangle_OBJECTS = \
"CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o" \
"CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o"

# External object files for target tutorial02_red_triangle
tutorial02_red_triangle_EXTERNAL_OBJECTS =

tutorial02_red_triangle: CMakeFiles/tutorial02_red_triangle.dir/tutorial02_red_triangle/tutorial02.cpp.o
tutorial02_red_triangle: CMakeFiles/tutorial02_red_triangle.dir/common/shader.cpp.o
tutorial02_red_triangle: CMakeFiles/tutorial02_red_triangle.dir/build.make
tutorial02_red_triangle: external/glfw-3.1.2/src/libglfw3.a
tutorial02_red_triangle: external/libGLEW_1130.a
tutorial02_red_triangle: CMakeFiles/tutorial02_red_triangle.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial09_vbo_indexing/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable tutorial02_red_triangle"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tutorial02_red_triangle.dir/link.txt --verbose=$(VERBOSE)
	/opt/homebrew/Cellar/cmake/3.27.7/bin/cmake -E copy /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/ogl-master\ 2/tutorial09_vbo_indexing/./tutorial02_red_triangle /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/ogl-master\ 2/tutorial02_red_triangle/

# Rule to build all files generated by this target.
CMakeFiles/tutorial02_red_triangle.dir/build: tutorial02_red_triangle
.PHONY : CMakeFiles/tutorial02_red_triangle.dir/build

CMakeFiles/tutorial02_red_triangle.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tutorial02_red_triangle.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tutorial02_red_triangle.dir/clean

CMakeFiles/tutorial02_red_triangle.dir/depend:
	cd "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial09_vbo_indexing" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2" "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2" "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial09_vbo_indexing" "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial09_vbo_indexing" "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial09_vbo_indexing/CMakeFiles/tutorial02_red_triangle.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/tutorial02_red_triangle.dir/depend

