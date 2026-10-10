// llm-tools.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

using namespace std;

void print_default()
{
	cout << "\nuse: llm-tools [src] [dst] [flags]\n"
		<< "\tsrc\t\t- path to file or folder (current by default) with .txt files\n"
		<< "\tdst\t\t- destination folder path (current by default)\n"
		<< "\t/norm\t\t- normalize whitespaces\n"
		<< "\t/dna\t\t- open Genome code file (file is ignored by default)\n"
		<< "\t/big\t\t- save combined and cleaned corpus to 'corpus.txt' file\n"
		<< "\t/gutten\t- strip Guttenberg Library head and tail\n"
		<< endl;
}

int main(int argc, char const* argv)
{
	/*
	* Open single source file
	* check if it is dna and should be ignored or used here
	* remove Guttenberg Library blocks (front end end) if have to
	* normalize if have to
	* add to resulting string
	* after last file save the result to 'corpus.txt' to folder (if applied)
	*/
	print_default();
}
