#include "stdio.h"
#include "wchar.h"
#include "malloc.h"
#include <sys/types.h>
#include <sys/stat.h>
typedef char* str;
typedef struct  {
    int a;
    void (*g)(str);
} foo;
typedef struct  {
    str name;
    int age;
} person;
str filename = "testcparser.lx";
int foo_f() {
    printf("hello");
    return 1;
}
int fileSize(str path) {
    struct _stat info;
    _stat(path, (&info));
    return (info . st_size);
}
int main() {
    FILE* f;
    int filesize;
    str buffer;
    (f = fopen(filename, "r"));
    if ((f == NULL)) {
        printf("Couldn't open file\n");
        exit(1);
    }
    (filesize = fileSize(filename));
    (buffer = malloc((filesize + 1)));
    if ((fread(buffer, 1, filesize, f) == ((-1)))) {
        printf("Couldn't read file\n");
        exit(1);
    }
    ((buffer[(filesize - 2)]) = 0);
    printf(buffer);
    if ((fclose(f) != 0)) {
        printf("Couldn't close file\n");
    }
}
__inline void foobaa(int a) {
    f();
}
