#include "common.h"

uint to_index(char c) {
    if (c == ' ') {
        return 26;
    } else if (c >= 'a' && c <= 'z') {
        return c - 'a';
    }
    return 27;
}

char to_char(uint index) {
    if (index == 26) {
        return ' ';
    } else if (index >= 0 && index < 26) {
        return 'a' + index;
    }
    return '?';
}