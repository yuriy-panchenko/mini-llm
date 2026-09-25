#include "pch.h"
#include "tokenizer.h"

namespace llm
{
	std::vector<Tokenizer::TokenId> Tokenizer::to_byte_ids(String const& text)
	{
		std::vector<Tokenizer::TokenId> ret;
		ret.reserve(text.length());

		for (auto ch : text)
			ret.push_back(static_cast<unsigned char>(ch));

		return ret;
	}

	void Tokenizer::train(String const& corpus, size_t vocabSize)
	{
		m_IdToBytes.reserve(vocabSize);
		m_IdToBytes.assign(kBaseVocabSize, {});
		for (size_t b = 0; b < kBaseVocabSize; ++b)
			m_IdToBytes[b] = String(1, static_cast<Char>(b));

		m_Merges.clear();
		m_MergeRank.clear();
		auto const rese{ vocabSize > kBaseVocabSize ? vocabSize - kBaseVocabSize : kBaseVocabSize };
		m_Merges.reserve(rese);
		m_MergeRank.reserve(rese);

		auto tokens{ to_byte_ids(corpus) };

		while (m_IdToBytes.size() < vocabSize)
		{
			// std::map (not unordered_map) so ties on count break deterministically,
			// on the lexicographically-smallest pair.
			std::map<Pair, size_t> counts;
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
			auto const newId{ m_IdToBytes.size() };

			m_IdToBytes.push_back(m_IdToBytes[pair.first] + m_IdToBytes[pair.second]);
			m_MergeRank[pair] = m_Merges.size();
			m_Merges.push_back(pair);
			tokens = merge(tokens, pair, newId);
		}
	}

	std::vector<Tokenizer::TokenId> Tokenizer::encode(String const& text) const
	{
		auto tokens{ to_byte_ids(text) };

		while (tokens.size() > 1)
		{
			size_t bestRank{ std::numeric_limits<size_t>::max() };

			for (size_t i = 0; i + 1 < tokens.size(); ++i)
			{
				auto const it{ m_MergeRank.find({ tokens[i], tokens[i + 1] }) };
				if (it != m_MergeRank.end() && it->second < bestRank)
					bestRank = it->second;
			}

			if (bestRank == std::numeric_limits<size_t>::max())
				break;   // no more learned merges apply to what's left

			tokens = merge(tokens, m_Merges[bestRank], kBaseVocabSize + bestRank);
		}

		return tokens;
	}

	Tokenizer::String Tokenizer::decode(std::vector<TokenId> const& ids) const
	{
		String out;

		for (auto id : ids)
			out += m_IdToBytes[id];

		return out;
	}

	std::vector<Tokenizer::TokenId> Tokenizer::merge(std::vector<TokenId>& tokens, Pair pair, size_t newId)
	{
		std::vector<TokenId> ret;
		ret.reserve(tokens.size());

		for (size_t i = 0; i < tokens.size(); ++i)
			if (i + 1 < tokens.size() && tokens[i] == pair.first && tokens[i + 1] == pair.second)
			{
				ret.push_back(newId);
				++i;
			}
			else ret.push_back(tokens[i]);
		return ret;
	}
}