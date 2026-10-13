#pragma once
#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// Pure text-preparation functions used by llm-tools (no state, no I/O).

// Strips CR, collapses a lone '\n' (mid-paragraph line wrap) to a space,
// collapses a run of 2+ '\n's (real paragraph break) down to one '\n'.
inline std::string normalize_whitespace(std::string const& text)
{
	std::string out;
	out.reserve(text.size());

	size_t runLen{ 0 };   // length of the '\n' run currently being counted

	for (char ch : text)
	{
		if (ch == '\r')
			continue;

		if (ch == '\n')
		{
			++runLen;
			continue;
		}

		if (runLen > 0)
		{
			out.push_back(runLen > 1 ? '\n' : ' ');
			runLen = 0;
		}
		out.push_back(ch);
	}

	if (runLen > 0)
		out.push_back(runLen > 1 ? '\n' : ' ');

	return out;
}

// Keeps only the book body between the START/END markers, in place.
// Returns true if a START (or, failing that, an END) marker was found. A file without markers
// (e.g. conversations.txt) is left unchanged and false is returned.
// Must run BEFORE normalize_whitespace: it looks for the end of the marker line.
inline bool strip_gutenberg(std::string& text)
{
	constexpr std::string_view startTag{ "*** START OF" }, endTag{ "*** END OF" };

	std::string out;
	size_t pos{ 0 };
	bool found{ false };
	while (true)
	{
		auto const s{ text.find(startTag, pos) };
		if (s == std::string::npos)
			break;
		auto const bodyStart{ text.find('\n', s) };
		if (bodyStart == std::string::npos)
			break;
		auto const e{ text.find(endTag, bodyStart) };
		auto const bodyEnd{ e == std::string::npos ? text.size() : e };   // truncated file: keep the rest

		auto first{ bodyStart + 1 };
		while (first < bodyEnd && std::isspace(static_cast<unsigned char>(text[first])))   // blank lines after the marker
			++first;

		out.append(text, first, bodyEnd - first);
		out += "\n\n";   // keep the last word of one book from fusing with the first of the next
		found = true;

		if (e == std::string::npos)
			break;
		pos = e + endTag.size();
	}

	if (found)
	{
		text = std::move(out);
		return true;
	}

	// No START (e.g. the head was already cut by BoilerplateIndex), but the END marker survived.
	auto const e{ text.find(endTag) };
	if (e == std::string::npos)
		return false;
	text.resize(e);
	return true;
}

// Share of digits among the non-blank characters, in percent (0 for an empty text).
// Prose sits well below 1%; tables, digit expansions and page-number runs are far above.
inline double digit_share(std::string const& text)
{
	size_t digits{ 0 }, chars{ 0 };
	for (char const ch : text)
	{
		if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\v' || ch == '\f')
			continue;
		++chars;
		if (ch >= '0' && ch <= '9')
			++digits;
	}
	return chars ? 100.0 * static_cast<double>(digits) / static_cast<double>(chars) : 0.0;
}

// Statistics of collapse_runs(), per symbol (the symbol itself as a UTF-8 string).
struct ComboStat
{
	size_t runs{ 0 };      // how many runs were shortened
	size_t removed{ 0 };   // how many symbols were cut away in total
	size_t longest{ 0 };   // the longest run seen (before the cut)
};
using ComboStats = std::map<std::string, ComboStat>;

// Length of the UTF-8 sequence at text[i] and its code point; {1, byte} for a stray/invalid byte.
inline std::pair<size_t, unsigned> utf8_at(std::string const& text, size_t i)
{
	auto const b0{ static_cast<unsigned char>(text[i]) };
	size_t len{ 1 };
	unsigned cp{ b0 };
	if (b0 >= 0xC2 && b0 <= 0xDF)
		len = 2, cp = b0 & 0x1Fu;
	else if (b0 >= 0xE0 && b0 <= 0xEF)
		len = 3, cp = b0 & 0x0Fu;
	else if (b0 >= 0xF0 && b0 <= 0xF4)
		len = 4, cp = b0 & 0x07u;
	if (len == 1 || i + len > text.size())
		return { 1, b0 };
	for (size_t k = 1; k < len; ++k)
	{
		auto const b{ static_cast<unsigned char>(text[i + k]) };
		if ((b & 0xC0u) != 0x80u)
			return { 1, b0 };
		cp = (cp << 6) | (b & 0x3Fu);
	}
	return { len, cp };
}

// "Special symbol": ASCII punctuation, or common typographic / drawing symbols
// (middle dot, general punctuation incl. dashes, bullets, ellipsis; box drawing, blocks, shapes).
// Letters, digits and whitespace never count, so "sssss" / "00000" / blank runs are never touched.
inline bool is_special_symbol(unsigned cp)
{
	if (cp < 0x80)
		return std::ispunct(static_cast<int>(cp)) != 0;
	return cp == 0xB7 || (cp >= 0x2010 && cp <= 0x205E) || (cp >= 0x2500 && cp <= 0x25FF);
}

// In place: every run of more than maxRun IDENTICAL special symbols ("------", "........",
// "=======") is cut down to exactly maxRun symbols. Returns the number of symbols removed.
// If stats is given, every shortened run is counted in it.
inline size_t collapse_runs(std::string& text, size_t maxRun, ComboStats* stats = nullptr)
{
	size_t r{ 0 }, w{ 0 }, removedTotal{ 0 };
	size_t const n{ text.size() };

	while (r < n)
	{
		// fast path: letters, digits, whitespace, UTF-8 text that is not a symbol
		auto const [len, cp] { utf8_at(text, r) };
		if (!is_special_symbol(cp))
		{
			if (w != r)
				std::memmove(&text[w], &text[r], len);
			w += len;
			r += len;
			continue;
		}

		// count how many times the same sequence repeats
		size_t count{ 1 };
		while (r + (count + 1) * len <= n && std::memcmp(&text[r], &text[r + count * len], len) == 0)
			++count;

		size_t const keep{ count < maxRun ? count : maxRun };
		if (count > maxRun && stats)   // before the move below can overwrite text[r]
		{
			auto& st{ (*stats)[text.substr(r, len)] };
			++st.runs;
			st.removed += count - maxRun;
			st.longest = std::max(st.longest, count);
		}
		if (w != r)
			std::memmove(&text[w], &text[r], keep * len);   // identical copies: the first 'keep' ones
		removedTotal += count - keep;
		w += keep * len;
		r += count * len;
	}

	text.resize(w);
	return removedTotal;
}

// Statistics of strip_prefixed(), per prefix.
struct StripStat
{
	size_t blocks{ 0 };   // how many places were cut
	size_t lines{ 0 };    // how many lines they spanned
	size_t chars{ 0 };    // how many bytes were removed
};
using StripStats = std::map<std::string, StripStat>;

// In place: drops every line that starts with one of the prefixes (leading blanks are ignored,
// matching is case sensitive). A "[Footnote 12: ... ]" often runs over several lines, so for a
// prefix that starts with '[' the block ends at the matching ']' (looked for in the next 40 lines;
// if there is none, only the line itself is dropped). Text after the closing ']' on the same
// line is kept. Returns the number of bytes removed.
inline size_t strip_prefixed(std::string& text, std::vector<std::string> const& prefixes, StripStats* stats = nullptr)
{
	constexpr size_t kLookahead{ 40 };
	size_t const n{ text.size() };
	std::string out;
	out.reserve(n);
	size_t pos{ 0 }, removed{ 0 };

	while (pos < n)
	{
		size_t const eol{ text.find('\n', pos) };
		size_t const lineEnd{ eol == std::string::npos ? n : eol + 1 };   // includes the '\n'

		size_t p{ pos };
		while (p < lineEnd && (text[p] == ' ' || text[p] == '\t'))
			++p;

		std::string const* hit{ nullptr };
		for (auto const& pre : prefixes)
			if (!pre.empty() && text.compare(p, pre.size(), pre) == 0)
			{
				hit = &pre;
				break;
			}

		if (!hit)
		{
			out.append(text, pos, lineEnd - pos);
			pos = lineEnd;
			continue;
		}

		size_t cutEnd{ lineEnd };
		if ((*hit)[0] == '[')
		{
			int depth{ 0 };
			size_t lines{ 0 }, q{ p };
			bool closed{ false };
			for (; q < n; ++q)
			{
				char const c{ text[q] };
				if (c == '[')
					++depth;
				else if (c == ']' && --depth == 0)
				{
					closed = true;
					break;
				}
				else if (c == '\n' && ++lines >= kLookahead)
					break;
			}
			if (closed)
			{
				cutEnd = q + 1;
				size_t t{ cutEnd };
				while (t < n && (text[t] == ' ' || text[t] == '\t'))
					++t;
				if (t < n && text[t] == '\r')
					++t;
				if (t >= n)
					cutEnd = n;
				else if (text[t] == '\n')
					cutEnd = t + 1;   // nothing else on the last line: take the line break too
				else if (t > cutEnd)
					cutEnd = t;       // text follows: keep it, without the blanks that separated it
			}
		}

		if (stats)
		{
			auto& st{ (*stats)[*hit] };
			++st.blocks;
			st.lines += static_cast<size_t>(std::count(text.begin() + pos, text.begin() + cutEnd, '\n')) + (text[cutEnd - 1] == '\n' ? 0 : 1);
			st.chars += cutEnd - pos;
		}
		removed += cutEnd - pos;
		pos = cutEnd;
	}

	text.swap(out);
	return removed;
}

// Long runs of ASCII digits found by long_digit_runs().
struct DigitRuns
{
	size_t runs{ 0 };      // how many runs of minLen+ digits
	size_t chars{ 0 };     // how many digits they hold in total
	size_t longest{ 0 };
	std::vector<std::string> samples;   // the first few runs (shortened), for the report
};

// In place: finds runs of at least minLen digits ("0503874373049"). With cut they are removed,
// together with one blank so no double space is left where the number stood.
inline DigitRuns long_digit_runs(std::string& text, size_t minLen, bool cut)
{
	constexpr size_t kSamples{ 3 }, kSampleLen{ 24 };
	DigitRuns res;
	size_t const n{ text.size() };
	size_t r{ 0 }, w{ 0 };

	while (r < n)
	{
		if (text[r] < '0' || text[r] > '9')
		{
			text[w++] = text[r++];
			continue;
		}

		size_t e{ r };
		while (e < n && text[e] >= '0' && text[e] <= '9')
			++e;
		size_t const len{ e - r };

		if (len < minLen)
		{
			std::memmove(&text[w], &text[r], len);
			w += len;
			r = e;
			continue;
		}

		++res.runs;
		res.chars += len;
		res.longest = std::max(res.longest, len);
		if (res.samples.size() < kSamples)
			res.samples.push_back(text.substr(r, std::min(len, kSampleLen)) + (len > kSampleLen ? "..." : ""));

		if (cut)
		{
			r = e;
			if (w > 0 && text[w - 1] == ' ')
			{
				if (r < n && text[r] == ' ')
					++r;
				else if (r >= n || text[r] == '\n' || text[r] == '\r')
					--w;
			}
		}
		else
		{
			std::memmove(&text[w], &text[r], len);
			w += len;
			r = e;
		}
	}

	text.resize(w);
	return res;
}

// Heuristic: does the file look like a raw nucleotide sequence? Looks at the first 64K
// non-blank characters (FASTA '>' / ';' header lines skipped) and requires >=95% of them
// to be uppercase A, C, G, T or N. Same alphabet extract_genome() works with: soft-masked
// (lowercase) sequences are NOT recognized and would be treated as text.
inline bool looks_like_dna(std::string const& text)
{
	constexpr size_t kSample{ 1 << 16 }, kMinChars{ 256 };

	size_t seen{ 0 }, bases{ 0 }, pos{ 0 };
	while (pos < text.size() && seen < kSample)
	{
		auto eol{ text.find('\n', pos) };
		if (eol == std::string::npos)
			eol = text.size();

		if (text[pos] != '>' && text[pos] != ';')
			for (size_t i{ pos }; i < eol && seen < kSample; ++i)
			{
				char const ch{ text[i] };
				if (ch == '\r' || ch == ' ' || ch == '\t')
					continue;
				++seen;
				if (ch == 'A' || ch == 'C' || ch == 'G' || ch == 'T' || ch == 'N')
					++bases;
			}

		pos = eol + 1;
	}

	return seen >= kMinChars && bases * 100 >= seen * 95;
}

// Extracts the first long run of A/C/G/T (line breaks ignored) from a genome file,
// dropping any header before it. Stops at the first foreign character after the run
// (an 'N' gap or a lowercase base ends the sequence).
inline std::string extract_genome(std::string const& src)
{
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

// ---------------------------------------------------------------------------
// Repeated head/tail detection (/auto)
//
// Every Gutenberg file starts and ends with licence text that is (almost) the same in
// hundreds of files, in several versions. Instead of listing the versions, look at the
// first/last kWindow bytes of ALL files: a line that shows up there in at least `minFiles`
// different files is boilerplate. Per file the head is cut after the last boilerplate line
// of the head window and the tail before the first boilerplate line of the tail window.
// Unique lines inside a block (title, author, "Produced by") are tolerated, up to kMaxGap
// in a row; the leftovers outside the block are unique, so they are harmless.
// ---------------------------------------------------------------------------
class BoilerplateIndex
{
public:
	static constexpr size_t kWindow{ 32 * 1024 };   // how much of each end is examined
	static constexpr size_t kMinLine{ 16 };         // shorter lines ("CHAPTER I.") are never evidence
	static constexpr size_t kMaxGap{ 10 };          // unique lines tolerated inside a boilerplate block

	struct Cut { size_t head{ 0 }, tail{ 0 }; };    // keep [head, tail)

	// Pass 1: feed the first and the last kWindow bytes of every file. `truncated` is true
	// when the file is longer than kWindow, i.e. the window edges cut a line in half.
	void add_file(std::string_view head, std::string_view tail, bool truncated)
	{
		std::vector<size_t> keys;

		auto collect = [&](std::string_view window, bool dropFirst, bool dropLast)
		{
			size_t pos{ 0 };
			bool first{ true };
			while (pos < window.size())
			{
				auto eol{ window.find('\n', pos) };
				bool const last{ eol == std::string_view::npos };
				if (last)
					eol = window.size();

				bool const partial{ (first && dropFirst) || (last && dropLast) };   // cut by the window edge
				first = false;
				if (!partial)
				{
					auto const line{ trim(window.substr(pos, eol - pos)) };
					if (line.size() >= kMinLine)
						keys.push_back(key(line));
				}
				pos = eol + 1;
			}
		};

		collect(head, false, truncated);   // the head window is cut at its end
		collect(tail, truncated, false);   // the tail window is cut at its start

		std::sort(keys.begin(), keys.end());
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());   // a file counts once per line
		for (auto const k : keys)
			++m_Df[k];
	}

	// Lines seen in at least minFiles files are boilerplate.
	void finish(size_t minFiles)
	{
		m_MinFiles = minFiles;
		m_Common = 0;
		for (auto const& kv : m_Df)
			if (kv.second >= minFiles)
				++m_Common;
	}

	size_t min_files() const { return m_MinFiles; }
	size_t boilerplate_lines() const { return m_Common; }

	// Pass 2: where to cut one whole file.
	Cut find_cut(std::string_view text) const
	{
		size_t const n{ text.size() };
		Cut cut{ 0, n };
		if (m_MinFiles == 0)
			return cut;
		bool const truncated{ n > kWindow };

		// Head: forward over the first kWindow bytes.
		{
			size_t const limit{ std::min(n, kWindow) };
			size_t pos{ 0 }, gap{ 0 };
			while (pos < limit)
			{
				auto eol{ text.find('\n', pos) };
				if (eol == std::string_view::npos)
					eol = n;
				if (truncated && eol >= limit)
					break;   // this line crosses the window edge

				auto const line{ trim(text.substr(pos, eol - pos)) };
				if (line.size() >= kMinLine)
				{
					if (common(line))
					{
						cut.head = std::min(eol + 1, n);
						gap = 0;
					}
					else if (++gap > kMaxGap)
						break;
				}
				pos = eol + 1;
			}
		}

		// Tail: backward over the last kWindow bytes, never into the head that was cut.
		{
			size_t const lowest{ std::max(cut.head, truncated ? n - kWindow : size_t{ 0 }) };
			size_t hi{ n }, gap{ 0 };
			if (hi > 0 && text[hi - 1] == '\n')
				--hi;   // terminator of the last line
			while (hi > lowest)
			{
				auto const nl{ text.rfind('\n', hi - 1) };
				size_t const start{ nl == std::string_view::npos ? 0 : nl + 1 };
				if (start < lowest)
					break;   // this line crosses the window edge

				auto const line{ trim(text.substr(start, hi - start)) };
				if (line.size() >= kMinLine)
				{
					if (common(line))
					{
						cut.tail = start;
						gap = 0;
					}
					else if (++gap > kMaxGap)
						break;
				}
				if (start == 0)
					break;
				hi = start - 1;
			}
		}

		if (cut.tail < cut.head)
			cut.tail = cut.head;
		return cut;
	}

private:
	static std::string_view trim(std::string_view s)
	{
		constexpr std::string_view ws{ " \t\r\v\f" };
		auto const a{ s.find_first_not_of(ws) };
		if (a == std::string_view::npos)
			return {};
		return s.substr(a, s.find_last_not_of(ws) - a + 1);
	}

	static size_t key(std::string_view line) { return std::hash<std::string_view>{}(line); }

	bool common(std::string_view line) const
	{
		auto const it{ m_Df.find(key(line)) };
		return it != m_Df.end() && it->second >= m_MinFiles;
	}

	std::unordered_map<size_t, size_t> m_Df;   // line hash -> number of files whose head/tail has it
	size_t m_MinFiles{ 0 }, m_Common{ 0 };
};
