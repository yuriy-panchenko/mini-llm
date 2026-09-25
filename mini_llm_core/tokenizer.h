#pragma once
//#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llm
{
	// Byte-level BPE tokenizer: trains merge rules on a corpus starting from the 256
	// raw byte values, then applies those merges (in learned priority order) to encode
	// new text. Operating on raw bytes rather than characters means decode(encode(x)) == x
	// for any input, including non-ASCII UTF-8, and there's never an "unknown token" —
	// worst case, unseen text just falls back to individual bytes.
	class Tokenizer
	{
		static constexpr size_t kBaseVocabSize{ 0x100 };

	public:
		using TokenId = size_t;
		using Char = char;
		using String = std::basic_string<Char>;

		// Learns merges from `corpus` until vocab_size() reaches `vocabSize`, or no
		// pair occurs more than once, whichever comes first. Resets any previous training.
		void train(String const& corpus, size_t vocabSize);

		std::vector<TokenId> encode(String const& text) const;
		String decode(std::vector<TokenId> const& ids) const;

		size_t vocab_size() const { return m_IdToBytes.size(); }

	private:
		using Pair = std::pair<TokenId, TokenId>;

		struct PairHash
		{
			size_t operator()(Pair const& p) const
			{
				return std::hash<TokenId>{}(p.first) ^ (std::hash<TokenId>{}(p.second) << 1);
			}
		};

		static std::vector<TokenId> to_byte_ids(String const& text);
		static std::vector<TokenId> merge(std::vector<TokenId>& tokens, Pair pair, size_t newId);

	private:
		std::vector<String> m_IdToBytes;                       // id -> byte sequence it expands to
		std::vector<Pair> m_Merges;                                 // merges, in the order they were learned
		std::unordered_map<Pair, size_t, PairHash> m_MergeRank;     // pair -> index into m_Merges (lower = merged earlier/preferred)
	};
}