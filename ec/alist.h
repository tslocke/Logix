typedef struct  {
    void** items;
    int len;
    int capacity;
}* alist;

void alist_index_error(alist l, int index);

void alist_increase_capacity(alist l);

__inline void* alist_item(alist l, int index);

__inline void alist_append(alist l, void* item);
