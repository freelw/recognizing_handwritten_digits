#include <iostream>

void testgrad();

int main(int argc, char *argv[]) {
    bool test = false;
    if (argc == 2) {
        if (std::string(argv[1]) == "test") {
            test = true;
        }
    }

    if (test) {
        testgrad();
    } else {

    }
    return 0;
}