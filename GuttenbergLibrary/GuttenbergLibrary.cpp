// GuttenbergLibrary.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <chrono>
#include "..\mini_llm_core\tokenizer.h"
#include "..\mini_llm_core\classmap.h"

constexpr auto source_root{ "..\\D184MB\\" };
//constexpr auto source_root{ "..\\D1GB\\" };
//constexpr auto source_root{ "..\\D1.7GB\\" };
//constexpr auto source_root{ "..\\D1GB\\Genome\\" };
constexpr auto destin_root{ "..\\Vocabs\\" };
//constexpr auto filename{ "vocab.tok" };

//constexpr size_t vocab_size{ 0x8000 };
constexpr size_t vocab_size{ 0xFFF0 };

using namespace std;
using namespace llm;
using namespace chrono;
namespace fs = filesystem;
//namespace clk = chrono;

Tokenizer gTok;

std::string GetGenome(std::ifstream& s)
{
	std::string const src{ istreambuf_iterator<char>{s}, istreambuf_iterator<char>{} };
	std::string ret;
	ret.reserve(src.length());
	bool found_start{ false };

	for (auto ch : src)
		switch (ch)
		{
		case 'A':
		case 'C':
		case 'G':
		case 'T':
			ret.push_back(ch);
			found_start = ret.size() > 256;
			break;
		case '\n':
		case '\r':
			break;
		default:
			if (found_start)
				return ret;
			else if (ret.size() < 256ull)
				ret.clear();
			break;
		}

	return ret;
}

void save_as(fs::path fn)
{
	cout << "\n\nSaving dictionary " << fn.filename() << "\t............\t";
	std::ofstream file{ fn, ios::binary };
	if (file)
	{
		file << gTok;
		cout << "OK!";
	}
	else cout << "ERR";

	cout << endl;
}


// Replaces every occurrence of `from` with `to`, in place.
// Handles overlapping/empty-replacement cases safely by advancing past each replacement.
// Usage:
// replace_all(text, "Bot: ", "");      // remove
// replace_all(text, "Bot: ", "A: ");   // replace
void replace_all(std::string& s, const std::string& from, const std::string& to)
{
	if (from.empty()) return;

	size_t pos = 0;
	while ((pos = s.find(from, pos)) != std::string::npos)
	{
		s.replace(pos, from.size(), to);
		pos += to.size();   // skip past what we just inserted
	}
}

std::string replace_all_fast(const std::string& s, const std::string& from, const std::string& to)
{
	if (from.empty()) return s;

	std::string out;
	out.reserve(s.size());

	size_t pos = 0, hit;
	while ((hit = s.find(from, pos)) != std::string::npos)
	{
		out.append(s, pos, hit - pos);   // copy the untouched stretch
		out += to;
		pos = hit + from.size();
	}
	out.append(s, pos, std::string::npos);  // tail
	return out;
}

int main()
{
	std::string corpus;
	//corpus = llm::read_all_files(source_root);
	//corpus = llm::normalize_whitespace(corpus);
	//std::ifstream fs{ std::string{source_root} + "2214.txt" };
	corpus = llm::read_file("..\\mini-llm\\conversations.txt");
	//replace_all(corpus, "User: ", "");
	//replace_all(corpus, "Bot: ", "");

	//tok.train(read_all_files(file_root), vocab_size);
	gTok.reset(vocab_size);
	gTok.tokenize(corpus);

	size_t milestone{ 0x200ull }, max_len{ 1ull }, best_count;
	std::string cmb_text;

	while (gTok.vocab_size() < vocab_size)
	{
		auto const tpStart{ steady_clock::now() };
		auto const best{ gTok.find_most_used_pair(&best_count) };
		if (best == Tokenizer::invalid_pair)
			break;
		auto tok_size{ gTok.get_tokens().size() };
		gTok.unite(best, best_count);
		auto const tpEnd{ steady_clock::now() };

		cmb_text = gTok.text(best);
		max_len = std::max(max_len, cmb_text.length());
		std::cout
			<< '\n' << tok_size
			<< ", lib " << gTok.vocab_size()
			<< ", max " << max_len
			<< '\t' << fixed << setprecision(3) << duration_cast<milliseconds>(tpEnd - tpStart).count() / 1000.
			<< '\t' << best_count << "\t\"" << cmb_text << '\"'
			;

		if (gTok.vocab_size() == milestone)
		{
			save_as(std::string{ destin_root } + "token" + (milestone >= 0x400 ? to_string(milestone / 0x400) + "K" : to_string(milestone)) + ".bin");
			milestone <<= 1;
		}
	}

	save_as(std::string{ destin_root } + "tokens.bin");
}
