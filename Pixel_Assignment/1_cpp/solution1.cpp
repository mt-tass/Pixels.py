#include <iostream>
#include <vector>

using namespace std;
namespace onedimension{
    void input(vector<int>& arr, int size){
        int num;
        for (int i = 0; i < size; i++){
            cout << "Enter Number:\n";
            cin >> num;
            arr.push_back(num);
        }
    }
    void print(vector<int>& arr){
        cout << "el at idx multiple of 3 :\n";
        for (int i = 0;i < arr.size(); i++){
            if (i % 3== 0){
                cout << arr[i] << "\n";
            }
        }
    }
}
namespace twodimension{
    void input(vector<vector<int>>& arr, int rows, int cols){
        int num;
        for (int i = 0; i < rows; i++){
            vector<int> temp;
            for (int j = 0; j < cols; j++){
                cout << "Enter Number:\n";
                cin >> num;
                temp.push_back(num);
            }
            arr.push_back(temp);
        }
    }

    void print(vector<vector<int>>& arr) {
        cout << "el at idx multiple of  3:\n";
        for (int i = 0; i < arr.size(); i++){
            for (int j = 0; j < arr[i].size(); j++){
                if (j % 3 == 0){
                    cout << arr[i][j] << endl;
                }
            }
        }
    }
}
int main() {
    vector<int> arr1D;
    vector<vector<int>> arr2D;

    cout << "enter elements of 2d :\n";
    twodimension::input(arr2D, 3, 4);
    cout << "enter elements of 1d :\n";
    onedimension::input(arr1D, 9);

    twodimension::print(arr2D);
    onedimension::print(arr1D);
}
