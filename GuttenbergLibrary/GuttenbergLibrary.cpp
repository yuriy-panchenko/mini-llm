// GuttenbergLibrary.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include "..\mini_llm_core\tokenizer.h"

constexpr auto file_root{ "..\\D184MB\\" };
constexpr auto filename{ "vocab.tok" };
constexpr size_t vocab_size{ 0x8000 };

using namespace std;
using namespace llm;
namespace fs = filesystem;

std::string read_file(fs::path const& filename)
{
	std::ifstream file{ filename, std::ios::binary };

	if (!file)
		throw std::runtime_error("Cannot open file: " + filename.string());

	return {
		std::istreambuf_iterator<char>(file),
		std::istreambuf_iterator<char>()
	};
}

string read_all_files(fs::path const& root)
{
	string ret;

	for (auto const& en : fs::directory_iterator{ root })
		if (en.is_regular_file() && en.path().extension() == ".txt")
		{
			string text;
			try
			{
				text = read_file(en.path());
			}
			catch (const std::exception&)
			{

			}
			ret.append(std::move(text));
		}
	return ret;
}

int main()
{
	Tokenizer tok;
	tok.train(read_all_files(file_root), vocab_size);
	std::ofstream file{ filename, ios::binary };
	if (file)
		file << tok;
}
