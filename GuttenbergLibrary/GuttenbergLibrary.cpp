// GuttenbergLibrary.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <chrono>
#include "..\mini_llm_core\tokenizer.h"

constexpr auto file_root{ "..\\D184MB\\" };
//constexpr auto filename{ "vocab.tok" };
constexpr size_t vocab_size{ 0x8000 };

using namespace std;
using namespace llm;
using namespace chrono;
namespace fs = filesystem;
//namespace clk = chrono;

Tokenizer gTok;

std::string read_file(fs::path const& filename)
{
	std::ifstream file{ filename, std::ios::binary };
	if (file)
		return { istreambuf_iterator<char>{file},istreambuf_iterator<char>{} };
	else throw std::runtime_error("Cannot open file: " + filename.string());

}

string read_all_files(fs::path const& root)
{
	string ret;

	cout << "Reading \'*.txt\' files in " << root << "\n\n";

	size_t file_count{};
	for (auto const& en : fs::directory_iterator{ root })
		if (en.is_regular_file() && en.path().extension() == ".txt")
		{
			string text;
			try
			{
				cout << file_count + 1 << " " << en.path().filename();
				text = read_file(en.path());
				++file_count;
				cout << '\t' << fs::file_size(en.path());

				if (file_count % 4)
					cout << '\t';
				else cout << '\n';
			}
			catch (const std::exception&)
			{
				cout << "\tFailed";
			}
			ret.append(std::move(text));
		}

	cout << "\n===============================================================================================\
		\nAll Files\t" << file_count
		<< "\nCharacters\t" << ret.length() << endl;

	return ret;
}

void save_as(fs::path fn)
{
	cout << "\nSaving dictionary " << fn.filename() << ".....";
	std::ofstream file{ fn, ios::binary };
	if (file)
	{
		file << gTok;
		cout << "OK!";
	}
	else cout << "ERR";

	cout << endl;
}

int main()
{
	auto const corpus{ read_all_files(file_root) };

	//tok.train(read_all_files(file_root), vocab_size);
	gTok.reset(vocab_size);

	auto tokens{ gTok.to_byte_ids(corpus) };
	size_t milestone{ 0x200ull }, max_len{ 1ull }, best_count;

	while (gTok.vocab_size() < vocab_size)
	{
		auto const tpStart{ steady_clock::now() };
		auto const best{ gTok.find_most_used_pair(tokens,&best_count) };
		if (best == Tokenizer::invalid_pair)
			break;
		tokens = gTok.unite(tokens, best);
		auto const tpEnd{ steady_clock::now() };

		{
			auto cmb_text{ gTok.text(best) };
			max_len = std::max(max_len, cmb_text.length());
			std::cout
				<< "\ntoken: " << tokens.size()
				<< ", lib: " << gTok.vocab_size()
				<< ", max_len: " << max_len
				<< '\t' << fixed << setprecision(3) << duration_cast<milliseconds>(tpEnd - tpStart).count() / 1000.
				<< '\t' << best_count << "\t\"" << cmb_text << "\""
				;
		}

		if (gTok.vocab_size() == milestone)
		{
			save_as("Vocabs\\token" + (milestone > 0x400 ? to_string(milestone / 0x400) + "K" : to_string(milestone)) + ".bin");
			milestone <<= 1;
		}
	}

	save_as("Vocabs\\token32K.bin");
}
