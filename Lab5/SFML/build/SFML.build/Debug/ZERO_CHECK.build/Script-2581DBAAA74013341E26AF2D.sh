#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/lab/Lab5/SFML"
  make -f /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/lab/Lab5/SFML/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "Release"; then :
  cd "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/lab/Lab5/SFML"
  make -f /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/lab/Lab5/SFML/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/lab/Lab5/SFML"
  make -f /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/lab/Lab5/SFML/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd "/Users/gongkaiwen/Desktop/研究生/第一学期/ece 6122/lab/Lab5/SFML"
  make -f /Users/gongkaiwen/Desktop/研究生/第一学期/ece\ 6122/lab/Lab5/SFML/CMakeScripts/ReRunCMake.make
fi

