#include <iostream>
using namespace std;
int main(){
    int x;
    int *p = &x;
    char *cp1 = (char *)p;
    char *cp2 = (char *)(p+1);
    int size = cp2-cp1;
    cout << size << "\n"; //prints 4 as size of int is 4
}