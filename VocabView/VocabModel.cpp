#include "pch.h"
#include "VocabModel.h"

// windows.h (via MFC pch) defines min/max macros; this file uses std::min/std::max.
#undef min
#undef max

namespace vv
{
	namespace
	{
		// Length of the valid UTF-8 sequence starting at s[i], or 0 if invalid there.
		size_t utf8_seq_len(std::string const& s, size_t i)
		{
			unsigned char const c{ (unsigned char)s[i] };
			size_t n{};
			if (c < 0x80) return 1;
			else if (c >= 0xC2 && c <= 0xDF) n = 2;
			else if (c >= 0xE0 && c <= 0xEF) n = 3;
			else if (c >= 0xF0 && c <= 0xF4) n = 4;
			else return 0;
			if (i + n > s.size()) return 0;
			for (size_t k = 1; k < n; ++k)
				if (((unsigned char)s[i + k] & 0xC0) != 0x80) return 0;
			unsigned char const c1{ (unsigned char)s[i + 1] };
			if (c == 0xE0 && c1 < 0xA0) return 0;		// overlong
			if (c == 0xED && c1 > 0x9F) return 0;		// surrogates
			if (c == 0xF0 && c1 < 0x90) return 0;		// overlong
			if (c == 0xF4 && c1 > 0x8F) return 0;		// > U+10FFFF
			return n;
		}

		std::string fmt(char const* f, double v) { char b[64]; snprintf(b, sizeof b, f, v); return b; }
		std::string num(size_t v) { return std::to_string(v); }
		std::string pct(size_t part, size_t whole) { return fmt("%.1f%%", whole ? 100.0 * part / whole : 0.0); }
	}

	std::string VocabModel::make_display(std::string const& raw, bool* pValid)
	{
		std::string out;
		bool valid{ true };
		for (size_t i = 0; i < raw.size();)
		{
			unsigned char const c{ (unsigned char)raw[i] };
			size_t const n{ utf8_seq_len(raw, i) };
			if (n == 0) { char b[8]; snprintf(b, sizeof b, "\\x%02X", c); out += b; valid = false; ++i; continue; }
			if (n == 1)
			{
				if (c == ' ') out += "\xC2\xB7";
				else if (c == '\n') out += "\\n";
				else if (c == '\r') out += "\\r";
				else if (c == '\t') out += "\\t";
				else if (c < 0x20 || c == 0x7F) { char b[8]; snprintf(b, sizeof b, "\\x%02X", c); out += b; }
				else out += (char)c;
			}
			else out.append(raw, i, n);
			i += n;
		}
		if (pValid) *pValid = valid;
		return out;
	}

	bool VocabModel::build(std::vector<std::string> const& vocab, std::vector<MergePair> const& merges, std::string* pError)
	{
		m_Tokens.clear(); m_Bytes.clear(); m_Order.clear(); m_Pos.clear(); m_Children.clear(); m_Inconsistent = 0;
		m_HasCorpus = false; m_CorpusTokens = m_CorpusBytes = 0; m_CorpusBadIds = 0;
		m_HasPairs = false; m_TopPairs.clear(); m_PairSum = 0; m_PairEntries = 0;
		auto fail = [&](char const* msg) { if (pError) *pError = msg; return false; };

		if (vocab.size() < kBaseVocab) return fail("Vocabulary has fewer than 256 tokens.");
		if (vocab.size() != kBaseVocab + merges.size()) return fail("Vocabulary size does not equal 256 + number of merges.");

		size_t const n{ vocab.size() };
		m_Bytes = vocab;
		m_Tokens.resize(n);
		m_Children.assign(n, {});

		for (uint32_t id = 0; id < n; ++id)
		{
			auto& t{ m_Tokens[id] };
			auto const& raw{ vocab[id] };
			t.id = id;
			t.byteLen = (uint32_t)raw.size();
			t.startsWithSpace = !raw.empty() && raw[0] == ' ';
			t.hasNewline = raw.find('\n') != std::string::npos;
			t.display = make_display(raw, &t.validUtf8);

			if (id >= kBaseVocab)
			{
				auto const& m{ merges[id - kBaseVocab] };
				t.rank = id - kBaseVocab;
				t.parentA = m.first;
				t.parentB = m.second;
				// A parent must already exist (id < this id); merged tokens are created in order.
				if (t.parentA >= id || t.parentB >= id) { t.consistent = false; t.depth = 0; }
				else
				{
					t.depth = 1 + std::max(m_Tokens[t.parentA].depth, m_Tokens[t.parentB].depth);
					t.consistent = (vocab[t.parentA] + vocab[t.parentB] == raw);
					m_Children[t.parentA].push_back(id);
					if (t.parentB != t.parentA) m_Children[t.parentB].push_back(id);
				}
				if (!t.consistent) ++m_Inconsistent;
			}
		}

		for (uint32_t id = 0; id < n; ++id)
			m_Tokens[id].uses = (uint32_t)m_Children[id].size();

		m_Order.resize(n);
		std::iota(m_Order.begin(), m_Order.end(), 0u);
		rebuild_positions();
		return true;
	}

	void VocabModel::rebuild_positions()
	{
		m_Pos.resize(m_Order.size());
		for (size_t row = 0; row < m_Order.size(); ++row)
			m_Pos[m_Order[row]] = (uint32_t)row;
	}

	void VocabModel::set_corpus(std::vector<llm::Tokenizer::TokenId> const& tokens)
	{
		m_HasCorpus = false; m_CorpusTokens = m_CorpusBytes = 0; m_CorpusBadIds = 0;
		for (auto& t : m_Tokens) t.freq = 0;
		if (tokens.empty() || m_Tokens.empty()) return;

		for (auto id : tokens)
		{
			if (id >= m_Tokens.size()) { ++m_CorpusBadIds; continue; }
			++m_Tokens[id].freq;
		}
		m_CorpusTokens = tokens.size();
		for (auto const& t : m_Tokens) m_CorpusBytes += t.freq * t.byteLen;
		m_HasCorpus = true;
	}

	void VocabModel::set_pairs(std::vector<PairCount> pairs)
	{
		m_HasPairs = false; m_TopPairs.clear(); m_PairSum = 0; m_PairEntries = pairs.size();
		if (pairs.empty()) return;
		for (auto const& p : pairs) m_PairSum += p.count;
		size_t const k{ std::min<size_t>(10, pairs.size()) };
		std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), [](PairCount const& a, PairCount const& b)
			{ return a.count != b.count ? a.count > b.count : a.pair < b.pair; });
		pairs.resize(k);
		m_TopPairs = std::move(pairs);
		m_HasPairs = true;
	}

	void VocabModel::sort(Column col, bool asc)
	{
		auto key = [&](uint32_t id) -> uint64_t
			{
				auto const& t{ m_Tokens[id] };
				switch (col)
				{
				case Column::Id:    return t.id;
				case Column::Rank:  return t.rank;			// kNone (base bytes) sorts last when ascending
				case Column::Bytes: return t.byteLen;
				case Column::Depth: return t.depth;
				case Column::Uses:  return t.uses;
				case Column::Freq:  return t.freq;
				default:            return 0;
				}
			};

		std::stable_sort(m_Order.begin(), m_Order.end(), [&](uint32_t a, uint32_t b)
			{
				int c{};
				if (col == Column::Text)
					c = m_Bytes[a].compare(m_Bytes[b]);
				else if (col == Column::Parents)
				{
					auto const& x{ m_Tokens[a] }; auto const& y{ m_Tokens[b] };
					if (x.parentA != y.parentA) c = x.parentA < y.parentA ? -1 : 1;
					else if (x.parentB != y.parentB) c = x.parentB < y.parentB ? -1 : 1;
				}
				else
				{
					auto const ka{ key(a) }, kb{ key(b) };
					c = ka < kb ? -1 : (ka > kb ? 1 : 0);
				}
				if (c == 0) return a < b;			// deterministic tie-break, always by id ascending
				return asc ? c < 0 : c > 0;
			});
		rebuild_positions();
	}

	std::vector<uint32_t> VocabModel::roots() const
	{
		std::vector<uint32_t> out(std::min<size_t>(kBaseVocab, m_Tokens.size()));
		std::iota(out.begin(), out.end(), 0u);
		std::sort(out.begin(), out.end(), [&](uint32_t a, uint32_t b) { return m_Pos[a] < m_Pos[b]; });
		return out;
	}

	std::vector<uint32_t> VocabModel::children(uint32_t id) const
	{
		if (id >= m_Children.size()) return {};
		auto out{ m_Children[id] };
		std::sort(out.begin(), out.end(), [&](uint32_t a, uint32_t b) { return m_Pos[a] < m_Pos[b]; });
		return out;
	}

	std::string VocabModel::tree(uint32_t id, size_t maxNodes) const
	{
		std::string out;
		size_t nodes{};
		auto rec = [&](auto& self, uint32_t t) -> void
			{
				auto const& ti{ m_Tokens[t] };
				// Valid parents always have a smaller id than the token they build. Anything else
				// (corrupt file) is shown as a leaf, which also rules out bad indices and cycles.
				bool const leaf{ ti.parentA == kNone || ti.parentA >= t || ti.parentB >= t };
				if (leaf || ++nodes > maxNodes)
				{
					out += leaf ? ti.display : "...";
					return;
				}
				out += "( ";
				self(self, ti.parentA);
				out += " + ";
				self(self, ti.parentB);
				out += " )";
			};
		if (id < m_Tokens.size()) rec(rec, id);
		return out;
	}

	uint32_t VocabModel::find(std::string const& needle, uint32_t startId) const
	{
		if (needle.empty() || m_Bytes.empty()) return kNone;
		size_t const n{ m_Bytes.size() };
		size_t const begin{ startId == kNone ? 0 : (size_t)startId + 1 };
		for (size_t k = 0; k < n; ++k)
		{
			size_t const id{ (begin + k) % n };
			if (m_Bytes[id].find(needle) != std::string::npos) return (uint32_t)id;
		}
		return kNone;
	}

	std::string VocabModel::report() const
	{
		if (m_Tokens.empty()) return "No vocabulary loaded.";
		std::string r;
		size_t const n{ m_Tokens.size() };
		size_t const merged{ n - kBaseVocab };

		std::vector<uint32_t> lens; lens.reserve(n);
		size_t totalBytes{}, utf8{}, spaceStart{}, newline{}, maxDepth{}, unused{};
		double depthSum{};
		for (auto const& t : m_Tokens)
		{
			lens.push_back(t.byteLen); totalBytes += t.byteLen;
			if (t.id >= kBaseVocab)
			{
				utf8 += t.validUtf8; spaceStart += t.startsWithSpace; newline += t.hasNewline;
				maxDepth = std::max<size_t>(maxDepth, t.depth); depthSum += t.depth;
				if (t.uses == 0) ++unused;
			}
		}
		std::sort(lens.begin(), lens.end());
		double const median{ (lens[(n - 1) / 2] + lens[n / 2]) / 2.0 };

		r += "Tokens: " + num(n) + "  (256 base bytes + " + num(merged) + " merges)\n";
		r += "Integrity: " + (m_Inconsistent ? num(m_Inconsistent) + " INCONSISTENT merged token(s)" : std::string{ "all merges consistent" }) + "\n\n";

		r += "Token length (bytes)\n";
		r += "  min " + num(lens.front()) + "   max " + num(lens.back()) + "   mean " + fmt("%.2f", (double)totalBytes / n) + "   median " + fmt("%.1f", median) + "\n";
		static constexpr struct { uint32_t lo, hi; char const* name; } kBuckets[]{
			{1,1,"1"},{2,2,"2"},{3,3,"3"},{4,4,"4"},{5,8,"5-8"},{9,16,"9-16"},{17,32,"17-32"},{33,~0u,"33+"} };
		size_t maxBucket{};
		std::vector<size_t> counts;
		for (auto const& b : kBuckets)
		{
			size_t c{};
			for (size_t i = kBaseVocab; i < n; ++i) c += (m_Tokens[i].byteLen >= b.lo && m_Tokens[i].byteLen <= b.hi);
			counts.push_back(c); maxBucket = std::max(maxBucket, c);
		}
		r += "  merged-token histogram:\n";
		for (size_t i = 0; i < counts.size(); ++i)
		{
			char line[96];
			snprintf(line, sizeof line, "  %6s | %-24s %zu\n", kBuckets[i].name,
				std::string(maxBucket ? counts[i] * 24 / maxBucket : 0, '#').c_str(), counts[i]);
			r += line;
		}

		r += "\nMerged tokens\n";
		r += "  merge depth: max " + num(maxDepth) + ", mean " + fmt("%.2f", merged ? depthSum / merged : 0) + "\n";
		r += "  valid UTF-8: " + num(utf8) + " (" + pct(utf8, merged) + "),  partial-char fragments: " + num(merged - utf8) + "\n";
		r += "  start with space: " + num(spaceStart) + " (" + pct(spaceStart, merged) + "),  contain newline: " + num(newline) + "\n";
		r += "  never used as a parent (leaf tokens): " + num(unused) + " (" + pct(unused, merged) + ")\n";

		auto top = [&](char const* title, auto keyFn)
			{
				std::vector<uint32_t> ids(n); std::iota(ids.begin(), ids.end(), 0u);
				size_t const k{ std::min<size_t>(5, n) };
				std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](uint32_t a, uint32_t b)
					{ auto ka{ keyFn(m_Tokens[a]) }, kb{ keyFn(m_Tokens[b]) }; return ka != kb ? ka > kb : a < b; });
				r += std::string{ "\n" } + title + "\n";
				for (size_t i = 0; i < k; ++i)
				{
					auto const& t{ m_Tokens[ids[i]] };
					std::string d{ t.display.size() > 40 ? t.display.substr(0, 40) + "..." : t.display };
					r += "  #" + num(t.id) + "  " + num(keyFn(t)) + "  \"" + d + "\"\n";
				}
			};
		top("Longest tokens (id, bytes)", [](TokenInfo const& t) { return (size_t)t.byteLen; });
		top("Most reused as a parent (id, merges)", [](TokenInfo const& t) { return (size_t)t.uses; });
		top("Deepest merge trees (id, depth)", [](TokenInfo const& t) { return (size_t)t.depth; });

		if (m_HasCorpus)
		{
			size_t neverSeen{};
			for (size_t i = kBaseVocab; i < n; ++i) neverSeen += (m_Tokens[i].freq == 0);
			r += "\nSaved token stream\n";
			r += "  " + num(m_CorpusTokens) + " tokens covering " + num(m_CorpusBytes) + " bytes  =>  "
				+ fmt("%.3f", m_CorpusTokens ? (double)m_CorpusBytes / m_CorpusTokens : 0.0) + " bytes/token\n";
			r += "  merged tokens absent from the stream: " + num(neverSeen) + " (" + pct(neverSeen, merged) + ")\n";
			if (m_CorpusBadIds) r += "  WARNING: " + num(m_CorpusBadIds) + " stream id(s) outside the vocabulary\n";
			top("Most frequent tokens (id, occurrences)", [](TokenInfo const& t) { return (size_t)t.freq; });
		}
		else
			r += "\nSaved token stream: none in this file (format v1 or not saved).\n";

		if (m_HasPairs)
		{
			r += "\nRemaining adjacent pairs: " + num(m_PairEntries) + " distinct, " + num(m_PairSum) + " occurrences";
			if (m_HasCorpus)
				r += (m_PairSum + 1 == m_CorpusTokens) ? "  (consistent with the stream)"
					: "  (MISMATCH: expected " + num(m_CorpusTokens ? m_CorpusTokens - 1 : 0) + ")";
			r += "\nNext merge candidates (count, pair):\n";
			for (auto const& pc : m_TopPairs)
			{
				if (pc.pair.first >= n || pc.pair.second >= n) continue;
				auto cut = [](std::string const& s) { return s.size() > 20 ? s.substr(0, 20) + "..." : s; };
				r += "  " + num(pc.count) + "  \"" + cut(m_Tokens[pc.pair.first].display) + "\" + \"" + cut(m_Tokens[pc.pair.second].display)
					+ "\"  (#" + num(pc.pair.first) + " + #" + num(pc.pair.second) + ")\n";
			}
		}

		// Duplicate byte strings would mean two ids decode identically -- should never happen in BPE.
		std::vector<uint32_t> ids(n); std::iota(ids.begin(), ids.end(), 0u);
		std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return m_Bytes[a] < m_Bytes[b]; });
		size_t dups{};
		for (size_t i = 1; i < n; ++i) dups += (m_Bytes[ids[i]] == m_Bytes[ids[i - 1]]);
		r += "\nDuplicate byte strings: " + num(dups) + (dups ? "  (unexpected)\n" : "\n");
		return r;
	}
}
