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
	class ISerialize abstract
	{
	public:
		friend std::ofstream& operator<<(std::ofstream& s, ISerialize& o) { o.Serialize(s); return s; }
		friend std::ifstream& operator>>(std::ifstream& s, ISerialize& o) { o.Serialize(s); return s; }
	private:
		virtual void Serialize(std::ofstream&) = 0;
		virtual void Serialize(std::ifstream&) = 0;
	};

	class Tokenizer
		:public ISerialize
	{
		static constexpr size_t kBaseVocabSize{ 0x100 };

	public:
		//using TokenId = size_t;
		using TokenId = unsigned short;
		using Char = char;
		using String = std::basic_string<Char>;
		static constexpr TokenId err_tok{ TokenId(~0x0) };

		// Learns merges from `corpus` until vocab_size() reaches `vocabSize`, or no
		// pair occurs more than once, whichever comes first. Resets any previous training.
		void train(String const& corpus, size_t vocabSize);

		std::vector<TokenId> encode(String const& text) const;
		String decode(std::vector<TokenId> const& ids) const;

		size_t vocab_size() const { return m_IdToBytes.size(); }

	public:
		using Pair = std::pair<TokenId, TokenId>;
		static constexpr Pair invalid_pair{ err_tok, err_tok };
		static Tokenizer::Pair find_most_used_pair(std::vector<Tokenizer::TokenId> const& tokens, size_t* pCount = nullptr);
		static std::vector<TokenId> to_byte_ids(String const& text);
		std::vector<Tokenizer::TokenId> unite(std::vector<Tokenizer::TokenId> const& tokens, Pair pair);
		void reset(size_t vocabSize);
		String text(Pair p)const { return m_IdToBytes[p.first] + m_IdToBytes[p.second]; }

	private:

		struct PairHash
		{
			size_t operator()(Pair const& p) const
			{
				return std::hash<TokenId>{}(p.first) ^ (std::hash<TokenId>{}(p.second) << 1);
			}
		};

		static std::vector<TokenId> merge(std::vector<TokenId> const& tokens, Pair pair, size_t newId);
		void Serialize(std::ofstream&) override;
		void Serialize(std::ifstream&) override;


	private:
		std::vector<String> m_IdToBytes;                       // id -> byte sequence it expands to
		std::vector<Pair> m_Merges;                                 // merges, in the order they were learned
		std::unordered_map<Pair, size_t, PairHash> m_MergeRank;     // pair -> index into m_Merges (lower = merged earlier/preferred)
	};
}