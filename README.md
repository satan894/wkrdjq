사용 허용합니다
회사 공개용 code
#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <cmath>
#include <bitset>
#include <fstream>

using namespace std;

int main() {
    // <vector> 사용: 숫자 벡터 생성
    vector<int>numbers = {1, 100, 1000, 10000, 100000, 1000000, 1000000, 10000000, };
    vector<double> sinValues(numbers.size());
    
 for (size_t i = 0;  i < numbers.size(); i++) {
        // <cmath> 사용: 삼각 함수 계산
        double radians = numbers[i] * (M_PI / 180.0); // 도(degree) -> 라디안 변환
        sinValues[i] = sin(radians);
    }
    
 // <string> 사용: 문자열 생성 및 출력
    string message = "AI 메모리 연산:";
    cout << message << endl;

 // <cctype> 사용: 메시지를 대문자로 변환
    for (char &ch : message) {
        if (isalpha(ch)) {
            ch = toupper(ch);
        }
    }
    cout << "대문자: " << message << endl;
    
 // sin 값 출력
    cout << "Sin 결과:" << endl;
    for (size_t i = 0; i < numbers.size(); i++) {
        cout << "sin(" << numbers[i] << "도) = " << sinValues[i] << endl;
    }
     ifstream inFile("bitset_output.txt");
    if (inFile.is_open()) {
        string bitsetStr;
        inFile >> bitsetStr; // 비트 집합을 문자열로 읽기
        
   // 비트 집합 크기 동적 처리
        bitset<64> bset(bitsetStr); // 최대 64비트까지 처리 가능
        cout << "파일에서 읽은 비트 집합: " << bset << endl;
        inFile.close();
    } else {
        cout << "commend" << endl;
    }
vector<int> numbers = {1,10,2,20,3,30,4,40,5,50};
for(size.s 1[i]<int>); /n'

return 0;
}

cd "/Users/satanhouse/Desktop/죽은 개발자/작업파일/" && g++ -std=c++20 AIsin.cpp -o AIsin && "/Users/satanhouse/Desktop/죽은 개발자/작업파일/"AIsin
satanhouse@SATAN-MacBook-Pro 작업파일 % cd "/Users/satanhouse/Desktop/죽은 개발자/작업파일/" && g++ -std=c++20 AIsin.cpp -o AIsin && "/Users/satanhouse/Desktop/죽은 개발자/작업파일/"AIsin
AI 메모리 연산:
대문자: AI 메모리 연산:
Sin 결과:
sin(1도) = 0.0174524
sin(100도) = 0.984808
sin(1000도) = -0.984808
sin(10000도) = -0.984808
sin(100000도) = -0.984808
sin(1000000도) = -0.984808
sin(1000000도) = -0.984808
sin(10000000도) = -0.984808
commend

