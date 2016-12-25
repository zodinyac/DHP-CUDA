#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define DEBUG_PRINT

// the number of a process topology dimensions
#define D 2

// enums for arrays
#define DEFINE_ENUM(dim1, dim2) \
    enum {                      \
        dim1 = 0,               \
        dim2 = 1                \
    }
DEFINE_ENUM(X, Y);
DEFINE_ENUM(DOWN, UP);
DEFINE_ENUM(LEFT, RIGHT);
DEFINE_ENUM(START, END);

#endif
