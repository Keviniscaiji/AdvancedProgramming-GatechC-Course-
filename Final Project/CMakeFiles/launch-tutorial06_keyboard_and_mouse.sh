#!/bin/sh
bindir=$(pwd)
cd /Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/ogl-master 2/tutorial06_keyboard_and_mouse/
export 

if test "x$1" = "x--debugger"; then
	shift
	if test "x" = "xYES"; then
		echo "r  " > $bindir/gdbscript
		echo "bt" >> $bindir/gdbscript
		GDB_COMMAND-NOTFOUND -batch -command=$bindir/gdbscript  /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/ogl-master\ 2/tutorial09_vbo_indexing/tutorial06_keyboard_and_mouse 
	else
		"/Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/ogl-master\ 2/tutorial09_vbo_indexing/tutorial06_keyboard_and_mouse"  
	fi
else
	"/Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/ogl-master\ 2/tutorial09_vbo_indexing/tutorial06_keyboard_and_mouse"  
fi
