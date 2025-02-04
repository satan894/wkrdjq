#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <cmath>
#include <bitset>
#include <fstream>

using namespace std;

int main() {
    // <vector> usage: Create a number vector
    vector<int> numbers = {1, 100, 1000, 10000, 100000, 1000000, 1000000, 10000000};
    vector<double> sinValues(numbers.size());

    for (size_t i = 0; i < numbers.size(); i++) {
        // <cmath> usage: Calculate trigonometric function
        double radians = numbers[i] * (M_PI / 180.0); // Convert degrees to radians
        sinValues[i] = sin(radians);
    }

    // <string> usage: Create and print a string
    string message = "AI Memory Operation:";
    cout << message << endl;

    // <cctype> usage: Convert message to uppercase
    for (char &ch : message) {
        if (isalpha(ch)) {
            ch = toupper(ch);
        }
    }
    cout << "Uppercase: " << message << endl;

    // Print sin values
    cout << "Sin Results:" << endl;
    for (size_t i = 0; i < numbers.size(); i++) {
        cout << "sin(" << numbers[i] << " degrees) = " << sinValues[i] << endl;
    }

    ifstream inFile("bitset_output.txt");
    if (inFile.is_open()) {
        string bitsetStr;
        inFile >> bitsetStr; // Read bitset as string

        // Dynamic bitset size handling
        bitset<64> bset(bitsetStr); // Supports up to 64 bits
        cout << "Bitset read from file: " << bset << endl;
        inFile.close();
    } else {
        cout << "Command" << endl;
    }

    vector<int> numbers2 = {1, 10, 2, 20, 3, 30, 4, 40, 5, 50};
    for (size_t i = 0; i < numbers2.size(); i++) {
        cout << numbers2[i] << " ";
    }
    cout << endl;

    return 0;
}
