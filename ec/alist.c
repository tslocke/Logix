#include "stdio.h"
#include "stdlib.h"
#include "general.h"
#include "alist.h"
void alist_index_error(alist l, int index) {
    printf("alist: index out of bounds: len=%d index=%d", ((l)->len), index);
    exit(1);
}
void alist_increase_capacity(alist l) {
    int newcap;
    (newcap = (((l)->capacity) * 2));
    (((l)->items) = xrealloc(((l)->items), newcap));
    (((l)->capacity) = newcap);
}
__inline void* alist_item(alist l, int index) {
    if ((index < ((l)->len))) {
        return (((l)->items)[index]);
    }
    else {
        alist_index_error(l, index);
    }
}
__inline void alist_append(alist l, void* item) {
    if ((((l)->len) == ((l)->capacity))) {
        alist_increase_capacity(l);
    }
    ((((l)->items)[((l)->len)]) = item);
    (((l)->len)++);
}
