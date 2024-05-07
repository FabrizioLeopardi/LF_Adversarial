#include <iostream>
#include <fstream>
using namespace std;

int main()
{
    fstream fin;
    fstream fout;
    
    //cout << "Run this code first" << endl;
    cout << "Which file should i transform?"  << endl;
    
    char x[100];
    for (int i=0; i<100; ++i)
        x[i] = '\0';
    
    cin >> x;
    
    fin.open(x, ios::in);
    fout.open("NN_data.txt", ios::out);
    
    char c;
    bool only_one_space=true;
    while(!fin.eof())
    {
        fin.get(c);
        if (c==' ' && !only_one_space)
        {
            fout << endl;
            only_one_space = true;
        }
        
        if (c=='\n' && !only_one_space)
        {
            fout << endl;
            only_one_space = true;
        }
        
        
        if (c=='0' || c=='1' || c=='2' || c=='3' || c=='4' || c=='5' || c=='6' || c=='7' || c=='8' || c=='9' || c=='.' || c=='-' || c=='e')
        {
            fout << c;
            only_one_space = false;
        }
    }
    
    fin.close();
    fout.close();
    return 0;
}
