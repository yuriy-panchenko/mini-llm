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
		static constexpr size_t kBaseVocabSize{ 256ull };

	public:
		using TokenId = size_t;

		// Learns merges from `corpus` until vocab_size() reaches `vocabSize`, or no
		// pair occurs more than once, whichever comes first. Resets any previous training.
		void train(std::string const& corpus, size_t vocabSize);

		std::vector<TokenId> encode(std::string const& text) const;
		std::string decode(std::vector<TokenId> const& ids) const;

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

		static std::vector<TokenId> to_byte_ids(std::string const& text);

	private:
		std::vector<std::string> m_IdToBytes;                       // id -> byte sequence it expands to
		std::vector<Pair> m_Merges;                                 // merges, in the order they were learned
		std::unordered_map<Pair, size_t, PairHash> m_MergeRank;     // pair -> index into m_Merges (lower = merged earlier/preferred)
	};
}