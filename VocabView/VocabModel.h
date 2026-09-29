#pragma once
// VocabModel: read-only, UI-independent view of a trained llm::Tokenizer's vocabulary.
// Pure std C++ (no MFC) so the statistics/sorting can be tested without a GUI.
namespace vv
{
	using MergePair = std::pair<llm::Tokenizer::TokenId, llm::Tokenizer::TokenId>;	// same as llm::Tokenizer::Pair

	constexpr uint32_t kNone{ 0xFFFFFFFFu };
	constexpr uint32_t kBaseVocab{ 256 };

	struct TokenInfo
	{
		uint32_t id{};
		uint32_t rank{ kNone };		// merge index (id - 256); kNone for the 256 base bytes
		uint32_t parentA{ kNone };
		uint32_t parentB{ kNone };
		uint32_t byteLen{};
		uint32_t depth{};			// merge-tree height: 0 for a base byte
		uint32_t uses{};			// how many merges use this token as a parent (= number of tree children)
		uint64_t freq{};			// occurrences in the saved token stream (0 if the file has none)
		bool     validUtf8{};
		bool     startsWithSpace{};
		bool     hasNewline{};
		bool     consistent{ true };	// vocab[id] == vocab[parentA] + vocab[parentB]
		std::string display;		// UTF-8, control bytes/invalid bytes escaped, ' ' shown as a middle dot
	};

	struct PairCount { MergePair pair; uint64_t count{}; };

	enum class Column { Id, Rank, Bytes, Depth, Uses, Freq, Parents, Text };

	class VocabModel
	{
	public:
		// Returns false (and leaves the model empty) if the data is structurally impossible,
		// e.g. fewer than 256 tokens, or vocab size != 256 + merges.size().
		bool build(std::vector<std::string> const& vocab, std::vector<MergePair> const& merges, std::string* pError = nullptr);

		size_t size() const { return m_Tokens.size(); }
		TokenInfo const& info(uint32_t id) const { return m_Tokens[id]; }
		std::string const& bytes(uint32_t id) const { return m_Bytes[id]; }

		// Display order. order()[row] is a token id.
		std::vector<uint32_t> const& order() const { return m_Order; }
		void sort(Column col, bool ascending);

		// Saved token stream (Tokenizer::get_tokens()): fills TokenInfo::freq and the corpus statistics.
		// Call after build(). Only counts are kept, not the stream itself.
		void set_corpus(std::vector<llm::Tokenizer::TokenId> const& tokens);
		// Saved adjacent-pair counts (Tokenizer::pair_counts()): the merges training would do next.
		void set_pairs(std::vector<PairCount> pairs);
		bool hasCorpus() const { return m_HasCorpus; }
		uint64_t corpusTokens() const { return m_CorpusTokens; }

		// Merge DAG, rooted at the 256 base bytes: the children of a token are the merges that use it
		// as a parent (so a token with two different parents appears under both). Both lists follow
		// the current sort order.
		std::vector<uint32_t> roots() const;
		std::vector<uint32_t> children(uint32_t id) const;
		// Full expansion down to base bytes, e.g. "(( t + h ) + e)".
		std::string tree(uint32_t id, size_t maxNodes = 64) const;

		// Substring search over raw bytes of every token, starting after `startId`, wrapping. kNone if not found.
		uint32_t find(std::string const& needle, uint32_t startId) const;

		std::string report() const;		// multi-line statistics text ('\n' separated)
		size_t inconsistentCount() const { return m_Inconsistent; }

		static std::string make_display(std::string const& raw, bool* pValidUtf8 = nullptr);

	private:
		std::vector<TokenInfo> m_Tokens;
		std::vector<std::string> m_Bytes;
		std::vector<uint32_t> m_Order;
		std::vector<uint32_t> m_Pos;						// id -> row in m_Order
		std::vector<std::vector<uint32_t>> m_Children;		// id -> merges built directly from it (id order)
		size_t m_Inconsistent{};

		bool m_HasCorpus{};
		uint64_t m_CorpusTokens{}, m_CorpusBytes{};
		size_t m_CorpusBadIds{};
		bool m_HasPairs{};
		std::vector<PairCount> m_TopPairs;					// most frequent remaining pairs
		uint64_t m_PairSum{};
		size_t m_PairEntries{};

		void rebuild_positions();
	};
}
