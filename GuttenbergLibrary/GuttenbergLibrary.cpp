// GuttenbergLibrary.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include "..\mini_llm_core\tokenizer.h"

constexpr auto file_root{ "..\\D184MB\\" };
//constexpr auto filename{ "vocab.tok" };
constexpr size_t vocab_size{ 0x8000 };

using namespace std;
using namespace llm;
namespace fs = filesystem;

Tokenizer tok;

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

	cout << "Reading \'*.txt\' files in " << root<<"\n\n";

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

	cout << "\n==========================================\
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
		file << tok;
		cout << "OK!";
	}
	else cout << "ERR";

	cout << endl;
}

int main()
{
	auto const corpus{ read_all_files(file_root) };

	//tok.train(read_all_files(file_root), vocab_size);
	tok.reset(vocab_size);

	auto tokens{ tok.to_byte_ids(corpus) };
	size_t milestone{ 0x200ull }, max_len{ 1ull }, best_count;

	while (tok.vocab_size() < vocab_size)
	{
		auto const best{ tok.find_most_used_pair(tokens,&best_count) };
		if (best == Tokenizer::invalid_pair)
			break;

		if (tok.vocab_size() == milestone)
		{
			save_as("token" + (milestone > 1000 ? to_string(milestone / 1000) + "K" : to_string(milestone)) + ".voc");
			milestone <<= 1;
		}

		auto cmb_text{ tok.text(best) };
		max_len = std::max(max_len, cmb_text.length());
		std::cout
			<< "\ntoken: " << tokens.size()
			<< ", lib: " << tok.vocab_size()
			<< ", max_len: " << max_len
			<< "\t" << best_count << "\t\"" << cmb_text << "\"";

		tokens = tok.unite(tokens, best);
	}

	save_as("token32K.voc");
}
