#include "pch.h"
#include "tokenizer.h"
#include <fstream>
#include <queue>

namespace llm
{
	std::vector<Tokenizer::TokenId> Tokenizer::to_byte_ids(String const& text)
	{
		/*std::vector<Tokenizer::TokenId> ret(text.length());
		auto itText{ text.begin() };
		for (auto& tok : ret)
			tok = static_cast<TokenId>(*itText++);
		return ret;*/
		std::vector<unsigned char> v{ text.begin(),text.end() };
		return { v.begin(),v.end() };
	}

	void Tokenizer::train(String const& corpus, size_t vocabSize)
	{
		reset(vocabSize);
		tokenize(corpus);

		while (m_IdToBytes.size() < vocabSize)
		{
			size_t occ{};
			auto const best{ find_most_used_pair(&occ) };
			if (best == invalid_pair)
				break;
			unite(best, occ);
		}
	}

	std::vector<Tokenizer::TokenId> Tokenizer::encode(String const& text) const
	{
		auto tok{ to_byte_ids(text) };
		auto const n{ tok.size() };
		if (n < 2)
			return tok;

		constexpr size_t null{ std::numeric_limits<size_t>::max() };

		std::vector<size_t> next(n), prev(n), version(n, 0);
		std::vector<bool> alive(n, true);
		for (size_t i = 0; i < n; ++i)
		{
			prev[i] = (i == 0) ? null : i - 1;
			next[i] = (i + 1 < n) ? i + 1 : null;
		}

		struct Entry
		{
			size_t rank, left, right, verLeft, verRight;
			bool operator>(Entry const& o) const { return rank > o.rank; }
		};
		std::priority_queue<Entry, std::vector<Entry>, std::greater<>> heap;

		auto tryPush = [&](size_t left)
			{
				if (left == null) return;
				auto const right{ next[left] };
				if (right == null) return;
				auto const it{ m_MergeRank.find({ tok[left], tok[right] }) };
				if (it != m_MergeRank.end())
					heap.push({ it->second, left, right, version[left], version[right] });
			};

		for (size_t i = 0; i < n; ++i)
			tryPush(i);

		while (!heap.empty())
		{
			auto const e{ heap.top() };
			heap.pop();

			if (!alive[e.left] || !alive[e.right] || next[e.left] != e.right
				|| version[e.left] != e.verLeft || version[e.right] != e.verRight)
				continue;   // stale -- one side changed since this was pushed

			tok[e.left] = TokenId(kBaseVocabSize + e.rank);   // e.left absorbs e.right
			++version[e.left];
			alive[e.right] = false;

			auto const after{ next[e.right] };
			next[e.left] = after;
			if (after != null)
				prev[after] = e.left;

			tryPush(prev[e.left]);   // left neighbour's pair now involves the new token
			tryPush(e.left);         // e.left's pair with its new right neighbour
		}

		std::vector<TokenId> out;
		out.reserve(n);
		for (size_t p = 0; p != null; p = next[p])
			out.push_back(tok[p]);

		return out;
	}

	Tokenizer::String Tokenizer::decode(std::vector<TokenId> const& ids) const
	{
		String out;

		for (auto id : ids)
			out += m_IdToBytes[id];

		return out;
	}

	std::vector<Tokenizer::TokenId> Tokenizer::merge(std::vector<TokenId> const& tokens, Pair pair, size_t newId)
	{
		std::vector<TokenId> ret;
		ret.reserve(tokens.size());

		for (size_t i = 0; i < tokens.size(); ++i)
			if (i + 1 < tokens.size() && tokens[i] == pair.first && tokens[i + 1] == pair.second)
			{
				ret.push_back(TokenId(newId));
				++i;
			}
			else ret.push_back(tokens[i]);
		return ret;
	}

	std::vector<Tokenizer::TokenId> Tokenizer::merge(std::vector<TokenId> const& tokens, Pair pair, size_t newId, std::vector<size_t>& tokenIdxs,
		std::vector<Pair>& decs)
	{
		std::vector<TokenId> ret;
		ret.reserve(tokens.size());

		TokenId const id{ TokenId(newId) };	//	constructed once, reused for every merged position
		bool skipLeftDec{ false };		//	true when the previous match's right-side lookahead already recorded this boundary

		for (size_t i = 0; i < tokens.size(); ++i)
			if (i + 1 < tokens.size() && tokens[i] == pair.first && tokens[i + 1] == pair.second)
			{
				if (i > 0 && !skipLeftDec)
					decs.push_back({ tokens[i - 1], tokens[i] });

				bool const adjacentNext{ i + 3 < tokens.size() && tokens[i + 2] == pair.first && tokens[i + 3] == pair.second };
				if (i + 2 < tokens.size())
					decs.push_back({ tokens[i + 1], tokens[i + 2] });
				skipLeftDec = adjacentNext;

				tokenIdxs.push_back(ret.size());
				ret.push_back(id);
				++i;
			}
			else ret.push_back(tokens[i]);

		return ret;
	}

	void Tokenizer::Serialize(std::ofstream& s)
	{
		unsigned __int64 u64{ m_IdToBytes.size() };
		s.write((char const*)&u64, sizeof u64);
		for (auto& str : m_IdToBytes)
		{
			unsigned short u16{ (unsigned short)str.length() };
			s.write((char const*)&u16, sizeof u16);
			s.write(str.data(), u16);
		}

		u64 = m_Merges.size();
		s.write((char const*)&u64, sizeof u64);
		s.write((char const*)m_Merges.data(), sizeof(Pair) * u64);

		u64 = m_MergeRank.size();
		s.write((char const*)&u64, sizeof u64);
		for (auto& item : m_MergeRank)
		{
			s.write((char const*)&item.first, sizeof Pair);
			s.write((char const*)&item.second, sizeof size_t);
		}
	}

	void Tokenizer::Serialize(std::ifstream& s)
	{
		m_IdToBytes.clear();
		m_MergeRank.clear();
		m_Merges.clear();

		unsigned __int64 u64;
		s.read((char*)&u64, sizeof u64);
		m_IdToBytes.resize(u64);
		unsigned short u16;
		for (auto& str : m_IdToBytes)
		{
			s.read((char*)&u16, sizeof u16);
			str.resize(u16);
			s.read(str.data(), u16);
		}

		s.read((char*)&u64, sizeof u64);
		m_Merges.resize(u64);
		s.read((char*)m_Merges.data(), sizeof(Pair) * u64);

		s.read((char*)&u64, sizeof u64);
		m_MergeRank.reserve(u64);
		Pair p;
		size_t sz;
		for (size_t i = 0; i < u64; i++)
		{
			s.read((char*)&p, sizeof Pair);
			s.read((char*)&sz, sizeof size_t);
			m_MergeRank[p] = sz;
		}
	}

	Tokenizer::Pair Tokenizer::find_most_used_pair(size_t* pCount)
	{
		if (!m_Counts.empty())
		{
			auto best{ m_Counts.begin() };

			for (auto it = m_Counts.begin(); it != m_Counts.end(); ++it)
				if (it->second > best->second)
					best = it;

			if (best->second > 1)
			{
				if (pCount)
					*pCount = best->second;
				return best->first;
			}
		}
		return invalid_pair;
	}

	void Tokenizer::unite(Pair pair, size_t occurance)
	{
		auto const newID{ m_IdToBytes.size() };
		m_IdToBytes.push_back(m_IdToBytes[pair.first] + m_IdToBytes[pair.second]);
		m_MergeRank[pair] = m_Merges.size();
		m_Merges.push_back(pair);

		auto& idx{ m_ScratchIdx };
		auto& decs{ m_ScratchDecs };
		idx.clear();
		decs.clear();
		idx.reserve(occurance);
		decs.reserve(2 * occurance);
		m_Tokens = merge(m_Tokens, pair, newID, idx, decs);

		auto const id{ TokenId(newID) };
		auto decrease = [this](Pair p)
			{
				assert(m_Counts.contains(p));
				--m_Counts[p];
				if (!m_Counts[p])
					m_Counts.erase(p);
			};

		for (auto& p : decs)
			decrease(p);

		// Dense marker instead of unordered_set<size_t>: mergedAt[pos] != 0 means
		// pos in the post-merge m_Tokens is a freshly-created id this round.
		auto& mergedAt{ m_ScratchMerged };
		mergedAt.assign(m_Tokens.size(), 0);
		for (auto ind : idx)
			mergedAt[ind] = 1;

		for (auto ind : idx)
		{
			if (ind + 1 < m_Tokens.size())	//	have element to the right
			{
				if (!mergedAt[ind + 1])
					++m_Counts[{ id, m_Tokens[ind + 1] }];
				else
					++m_Counts[{ id, id }];	//	count the new {id,id} edge exactly once
			}
			if (ind && !mergedAt[ind - 1])	//	have element to the left
				++m_Counts[{m_Tokens[ind - 1], id}];
		}

		m_Counts.erase(pair);

		auto count_all = [this]()->size_t
			{
				size_t ret{};
				for (auto& item : m_Counts)
					ret += item.second;
				return ret;
			};
		assert(count_all() == m_Tokens.size() - 1);
	}

	void Tokenizer::reset(size_t vocabSize)
	{
		m_Merges.clear();
		m_MergeRank.clear();
		m_Counts.clear();
		m_Tokens.clear();

		auto const rese{ vocabSize > kBaseVocabSize ? vocabSize - kBaseVocabSize : kBaseVocabSize };
		m_Merges.reserve(rese);
		m_MergeRank.reserve(rese);
		m_IdToBytes.reserve(vocabSize);
		m_IdToBytes.assign(kBaseVocabSize, {});
		//m_Counts.reserve(vocabSize);

		for (size_t b = 0; b < kBaseVocabSize; ++b)
			m_IdToBytes[b] = String(1, static_cast<Char>(b));
	}

	void Tokenizer::tokenize(String const& corpus)
	{
		m_Tokens = to_byte_ids(corpus);
		m_Counts.clear();

		for (size_t i = 0; i + 1 < m_Tokens.size(); ++i)
			++m_Counts[{ m_Tokens[i], m_Tokens[i + 1] }];
	}
}