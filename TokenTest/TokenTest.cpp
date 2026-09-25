// TokenTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <cassert>

using namespace std;
constexpr size_t kBaseVocabSize{ 0x100 };
using Char = char;
using String = std::basic_string<Char>;
using TokenId = size_t;
using Pair = pair<TokenId, TokenId>;
vector<String> IdToBytes;                       // id -> byte sequence it expands to
vector<Pair> Merges;                                 // merges, in the order they were learned
struct PairHash
{
	size_t operator()(Pair const& p) const
	{
		return std::hash<TokenId>{}(p.first) ^ (std::hash<TokenId>{}(p.second) << 1);
	}
};
unordered_map<Pair, size_t, PairHash> MergeRank;     // pair -> index into m_Merges (lower = merged earlier/preferred)

void train(String const& corpus, size_t vocabSize)
{
	IdToBytes.assign(kBaseVocabSize, {});
	for (size_t b = 0; b < kBaseVocabSize; ++b)
		IdToBytes[b] = String(1, static_cast<Char>(b));

	Merges.clear();
	MergeRank.clear();

	//vector<TokenId> tokens{ corpus.begin(), corpus.end() };
	vector<TokenId> tokens;
	tokens.reserve(corpus.size());
	for (unsigned char ch : corpus)
		tokens.push_back(ch);

	while (IdToBytes.size() < vocabSize)
	{
		// std::map (not unordered_map) so ties on count break deterministically,
		// on the lexicographically-smallest pair.
		map<Pair, size_t> counts;
		for (size_t i = 0; i + 1 < tokens.size(); ++i)
			++counts[{ tokens[i], tokens[i + 1] }];

		if (counts.empty())
			break;

		auto best{ counts.begin() };
		for (auto it = counts.begin(); it != counts.end(); ++it)
			if (it->second > best->second)
				best = it;

		if (best->second < 2)
			break;   // no pair repeats — nothing left worth merging

		auto const pair{ best->first };
		auto const newId{ IdToBytes.size() };

		IdToBytes.push_back(IdToBytes[pair.first] + IdToBytes[pair.second]);
		MergeRank[pair] = Merges.size();
		Merges.push_back(pair);

		vector<TokenId> merged;
		merged.reserve(tokens.size());

		for (size_t i = 0; i < tokens.size(); ++i)
			if (i + 1 < tokens.size() && tokens[i] == pair.first && tokens[i + 1] == pair.second)
			{
				merged.push_back(newId);
				++i;
			}
			else merged.push_back(tokens[i]);

		tokens = std::move(merged);
	}
}

int main()
{
	/*std::wifstream file("..\\mini-llm\\text.txt");
	file.imbue(std::locale(file.getloc(), new std::codecvt_utf8<wchar_t>));
	std::wstring text{ std::istreambuf_iterator<wchar_t>(file), std::istreambuf_iterator<wchar_t>() };*/
	std::ifstream file("..\\mini-llm\\text.txt");
	String text{ std::istreambuf_iterator<Char>(file), std::istreambuf_iterator<Char>() };

	train(text, kBaseVocabSize << 2);
}
