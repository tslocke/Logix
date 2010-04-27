#include "stdio.h"
#include "stdlib.h"
#include "general.h"
void fatal(str message) {
    printf("%s\n", message);
}
void* xrealloc(void* ptr, size_t size) {
    void* p;
    (p = realloc(ptr, size));
    if ((p == 0)) {
        fatal("Could not allocate memory");
    }
    return p;
}
