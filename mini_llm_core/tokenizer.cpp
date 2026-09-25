#include "pch.h"
#include "tokenizer.h"

namespace llm
{
	std::vector<Tokenizer::TokenId> Tokenizer::to_byte_ids(String const& text)
	{
		return { text.begin(), text.end() };
	}

	void Tokenizer::train(String const& corpus, size_t vocabSize)
	{
		m_IdToBytes.assign(kBaseVocabSize, std::wstring{});
		for (size_t b = 0; b < kBaseVocabSize; ++b)
			m_IdToBytes[b] = std::wstring{ 1, static_cast<wchar_t>(b) };

		m_Merges.clear();
		m_MergeRank.clear();

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

			std::vector<TokenId> merged;
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

			auto const pair{ m_Merges[bestRank] };
			auto const newId{ kBaseVocabSize + bestRank };

			std::vector<TokenId> merged;
			merged.reserve(tokens.size());

			for (size_t i = 0; i < tokens.size();)
				if (i + 1 < tokens.size() && tokens[i] == pair.first && tokens[i + 1] == pair.second)
				{
					merged.push_back(newId);
					i += 2;
				}
				else
				{
					merged.push_back(tokens[i]);
					++i;
				}

			tokens = std::move(merged);
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
}