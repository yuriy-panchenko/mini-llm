#pragma once
#include <algorithm>
#include <cctype>
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
