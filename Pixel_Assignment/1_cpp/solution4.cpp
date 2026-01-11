#include <iostream>
#include <vector>
using namespace std;
int main(){
    vector<int> arr;
    int size;
    cout << "Enter the size of array : \n";
    cin >> size;
    cout << "enter the elements of array : \n";
    for(int i = 0 ; i < size ; i++){
        int num;
        cin >> num;
        arr.push_back(num);
    }
    int max = INT8_MIN;
    int min = INT8_MAX;
    for(int i = 0 ; i < size ; i++){
        if(arr[i] > max){
            max = arr[i];
        }
        if(arr[i] < min){
            min = arr[i];
        }
    }
    cout << "max : " << max << endl;
    cout << "min : " << min << endl;
}