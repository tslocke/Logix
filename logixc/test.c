#include "stdio.h"

typedef enum {a, b, c} foo;
typedef void* ptr;


typedef struct P {
  char* name;
  struct P* friends;
  int (*f)(int, int);
} person;


int main() {
  struct foo {
    int x;
    int y;
  } a;

  int* (*b[2])(int);

}

__inline void fog() {
  person* p;
  int index = 0;
  printf("alist: index out of bounds: len=%d index=%d", ((p)->name), index);
}
