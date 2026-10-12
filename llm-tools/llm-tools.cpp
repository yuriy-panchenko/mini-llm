// llm-tools.cpp : prepares a text corpus (for gutlib / training).
//
// Per source file: genome files are ignored (or extracted with /dna), repeated head/tail
// blocks are cut (/auto), Gutenberg head/tail is stripped (/gutten), whitespace is
// normalized (/norm), long runs of one symbol are shortened (/combo), and the result is appended to the combined corpus, saved as
// 'corpus.txt' with /big.
//

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "text_ops.h"

using namespace std;
namespace fs = filesystem;

struct Options
{
	bool norm{ false }, dna{ false }, big{ false }, gutten{ false }, autoCut{ false };
	double maxDigits{ 0 };   // /maxdigits=N: skip files with more than N% digits (0 = off)
	size_t autoMin{ 0 };   // /auto=N: a line must repeat in N files to be boilerplate (0 = automatic)
	bool combo{ false };
	size_t comboMax{ 3 };   // /combo[=N]: runs of one symbol are cut down to N symbols
};

void print_default()
{
	cout << "\nuse: llm-tools [src] [dst] [flags]\n"
		<< "\tsrc\t\t- path to file or folder (current by default) with .txt files\n"
		<< "\tdst\t\t- destination folder path (current by default)\n"
		<< "\t/norm\t\t- normalize whitespaces\n"
		<< "\t/dna\t\t- open Genome code file (file is ignored by default)\n"
		<< "\t/big\t\t- save combined and cleaned corpus to 'corpus.txt' file\n"
		<< "\t/gutten\t- strip Guttenberg Library head and tail\n"
		<< "\t/auto[=N]\t- find head/tail blocks repeated across files (any version) and cut them;\n"
		<< "\t\t\t  N = in how many files a line must repeat (default 1% of files, at least 3)\n"
		<< "\t/combo[=N]\t- shorten runs of more than N (default 3) identical special symbols\n"
		<< "\t\t\t  ('------', '........', '=======', ...) to N symbols; letters, digits and\n"
		<< "\t\t\t  spaces are never touched; a report per symbol is printed at the end\n"
		<< "\t/maxdigits=N\t- skip files with more than N% digits (tables, number runs); the share is\n"
		<< "\t\t\t  shown for every file\n"
		<< endl;
}

// "12.3%"
string pct_str(double pct)
{
	char buf[32];
	snprintf(buf, sizeof buf, "%.1f%%", pct);
	return buf;
}

// Binary read: no newline translation, so what is processed is what is on disk.
string read_file(fs::path const& fn)
{
	ifstream file{ fn, ios::binary };
	if (!file)
		throw runtime_error("Cannot open file: " + fn.string());
	return { istreambuf_iterator<char>{file}, istreambuf_iterator<char>{} };
}

// Pass 1 of /auto: only the first and the last kWindow bytes of a file, the middle is not read.
void read_edges(fs::path const& fn, string& head, string& tail, bool& truncated)
{
	ifstream file{ fn, ios::binary };
	if (!file)
		throw runtime_error("Cannot open file: " + fn.string());

	auto const size{ fs::file_size(fn) };
	auto const n{ static_cast<size_t>(min<uintmax_t>(size, BoilerplateIndex::kWindow)) };

	head.resize(n);
	file.read(head.data(), static_cast<streamsize>(n));
	head.resize(static_cast<size_t>(file.gcount()));

	truncated = size > n;
	if (!truncated)
	{
		tail = head;
		return;
	}

	tail.resize(n);
	file.seekg(static_cast<streamoff>(size - n));
	file.read(tail.data(), static_cast<streamsize>(n));
	tail.resize(static_cast<size_t>(file.gcount()));
}

// "-100500" / "+5" / "0": how much a file lost (or gained) in preparation.
string delta_str(size_t before, size_t after)
{
	if (after == before)
		return "0";
	return after < before ? "-" + to_string(before - after) : "+" + to_string(after - before);
}

// "'.'" for printable ASCII, "U+2026 '…'" for the rest.
string symbol_str(string const& sym)
{
	auto const [len, cp] { utf8_at(sym, 0) };
	if (len == 1 && cp < 0x80)
		return string{ "'" } + sym + "'";
	char buf[16];
	snprintf(buf, sizeof buf, "U+%04X", cp);
	return string{ buf } + " '" + sym + "'";
}

// End report of /combo: which symbols were shortened, most removed first.
void print_combo_report(ComboStats const& stats, size_t maxRun)
{
	cout << "\n=== Combo report (runs longer than " << maxRun << " cut to " << maxRun << ") ===\n";
	if (stats.empty())
	{
		cout << "no runs found\n";
		return;
	}

	vector<pair<string, ComboStat>> rows{ stats.begin(), stats.end() };
	sort(rows.begin(), rows.end(), [](auto const& a, auto const& b)
		{
			return a.second.removed != b.second.removed ? a.second.removed > b.second.removed : a.first < b.first;
		});

	size_t runs{ 0 }, removed{ 0 };
	for (auto const& [sym, st] : rows)
	{
		cout << "\t" << symbol_str(sym) << "\truns " << st.runs << "\tremoved " << st.removed << "\tlongest " << st.longest << '\n';
		runs += st.runs;
		removed += st.removed;
	}
	cout << "\ttotal\t" << rows.size() << " symbols, runs " << runs << ", removed " << removed << "\n";
}

// A folder gives its .txt files in a fixed (sorted) order, a file gives itself.
vector<fs::path> collect_files(fs::path const& src)
{
	vector<fs::path> files;
	if (fs::is_directory(src))
	{
		for (auto const& en : fs::directory_iterator{ src })
			if (en.is_regular_file() && en.path().extension() == ".txt")
				files.push_back(en.path());
		sort(files.begin(), files.end());   // directory_iterator order is unspecified
	}
	else
		files.push_back(src);
	return files;
}

int main(int argc, char const* argv[])
{
	if (argc == 1)
	{
		print_default();
		return EXIT_SUCCESS;
	}

	// ------------------------------------------------------------------
	// Command-line parsing
	// ------------------------------------------------------------------
	Options opt;
	vector<string> positional;
	for (int i = 1; i < argc; ++i)
	{
		string const arg{ argv[i] };
		if (arg.empty())
			continue;

		if (arg.front() == '/' || arg.front() == '-')
		{
			string flag{ next(arg.begin()), arg.end() };
			transform(flag.begin(), flag.end(), flag.begin(), [](unsigned char a) { return static_cast<char>(tolower(a)); });
			if (flag == "norm")
				opt.norm = true;
			else if (flag == "dna")
				opt.dna = true;
			else if (flag == "big")
				opt.big = true;
			else if (flag == "gutten")
				opt.gutten = true;
			else if (flag.rfind("maxdigits=", 0) == 0)
			{
				try
				{
					opt.maxDigits = stod(flag.substr(10));
				}
				catch (...)
				{
					opt.maxDigits = 0;
				}
				if (!(opt.maxDigits > 0 && opt.maxDigits <= 100))
				{
					cerr << "Error: /maxdigits needs a value in (0, 100]: " << arg << endl;
					return EXIT_FAILURE;
				}
			}
			else if (flag == "combo")
				opt.combo = true;
			else if (flag.rfind("combo=", 0) == 0)
			{
				try
				{
					opt.comboMax = stoull(flag.substr(6));
				}
				catch (...)
				{
					opt.comboMax = 0;
				}
				if (opt.comboMax == 0)
				{
					cerr << "Error: /combo needs a value of 1 or more: " << arg << endl;
					return EXIT_FAILURE;
				}
				opt.combo = true;
			}
			else if (flag == "auto")
				opt.autoCut = true;
			else if (flag.rfind("auto=", 0) == 0)
			{
				try
				{
					opt.autoMin = stoull(flag.substr(5));
				}
				catch (...)
				{
					opt.autoMin = 0;
				}
				if (opt.autoMin == 0)
				{
					cerr << "Error: invalid /auto value: " << arg << endl;
					return EXIT_FAILURE;
				}
				opt.autoCut = true;
			}
			else if (flag == "?" || flag == "h" || flag == "help")
			{
				print_default();
				return EXIT_SUCCESS;
			}
			else
			{
				cerr << "Unknown flag: " << arg << endl;
				return EXIT_FAILURE;
			}
		}
		else
			positional.push_back(arg);
	}

	if (positional.size() > 2)
	{
		cerr << "Error: too many arguments (expected [src] [dst] [flags]).\n";
		return EXIT_FAILURE;
	}

	fs::path const src{ positional.size() >= 1 ? fs::path{ positional[0] } : fs::current_path() };
	fs::path const dst{ positional.size() >= 2 ? fs::path{ positional[1] } : fs::current_path() };
	fs::path const corpusPath{ dst / "corpus.txt" };

	if (!fs::exists(src))
	{
		cerr << "Error: source path does not exist: " << src << endl;
		return EXIT_FAILURE;
	}

	cout << "=== llm-tools ===\n"
		<< "Source      : " << src << "\n"
		<< "Destination : " << dst << "\n"
		<< "Normalize   : " << (opt.norm ? "yes" : "no") << "\n"
		<< "DNA files   : " << (opt.dna ? "used" : "ignored") << "\n"
		<< "Del Gutten  : " << (opt.gutten ? "yes" : "no") << "\n"
		<< "Auto head/tail: " << (opt.autoCut ? "yes" : "no") << "\n"
		<< "Combo runs  : " << (opt.combo ? "max " + to_string(opt.comboMax) + " in a row" : "off") << "\n"
		<< "Max digits  : " << (opt.maxDigits > 0 ? pct_str(opt.maxDigits) : "off") << "\n"
		<< "Save corpus : " << (opt.big ? "yes" : "no") << "\n"
		<< "=================\n" << endl;

	auto files{ collect_files(src) };

	// A corpus.txt left by a previous run must not become an input of this one.
	size_t skippedOld{ 0 };
	if (opt.big && fs::exists(corpusPath))
		skippedOld = erase_if(files, [&](fs::path const& f) { return fs::equivalent(f, corpusPath); });

	// ------------------------------------------------------------------
	// Pass 1 (/auto): which head/tail lines repeat across the files?
	// ------------------------------------------------------------------
	BoilerplateIndex boiler;
	if (opt.autoCut)
	{
		if (files.size() < 10)
		{
			cout << "Auto head/tail needs at least 10 files to tell boilerplate from text, skipped.\n" << endl;
			opt.autoCut = false;
		}
		else
		{
			cout << "Scanning the head and tail of " << files.size() << " files...\n";
			string head, tail;
			bool truncated{ false };
			for (auto const& f : files)
				try
				{
					read_edges(f, head, tail, truncated);
					boiler.add_file(head, tail, truncated);
				}
				catch (exception const&)
				{
					// reported in pass 2
				}
			boiler.finish(opt.autoMin ? opt.autoMin : max<size_t>(3, files.size() / 100));
			cout << boiler.boilerplate_lines() << " repeated lines (in at least " << boiler.min_files() << " files)\n" << endl;
		}
	}

	ofstream out;
	if (opt.big)
	{
		fs::create_directories(dst);
		out.open(corpusPath, ios::binary | ios::trunc);
		if (!out)
		{
			cerr << "Error: cannot write " << corpusPath << endl;
			return EXIT_FAILURE;
		}
	}

	// ------------------------------------------------------------------
	// One file at a time: open -> dna? -> strip Gutenberg -> normalize -> append
	// ------------------------------------------------------------------
	size_t index{ 0 }, used{ 0 }, failed{ 0 }, dnaIgnored{ 0 }, noMarkers{ 0 }, emptied{ 0 };
	size_t charsIn{ 0 }, charsOut{ 0 };
	size_t digitSkipped{ 0 }, digitSkippedChars{ 0 };
	vector<pair<fs::path, double>> digitList;
	ComboStats comboTotal;
	size_t comboFiles{ 0 }, comboChars{ 0 };
	size_t autoHeadFiles{ 0 }, autoTailFiles{ 0 }, autoHeadChars{ 0 }, autoTailChars{ 0 };

	for (auto const& f : files)
	{
		cout << ++index << '\t' << f.filename();

		string text;
		try
		{
			text = read_file(f);
		}
		catch (exception const&)
		{
			cout << "\tFailed\n";
			++failed;
			continue;
		}
		auto const sizeIn{ text.size() };

		if (text.empty())
		{
			cout << "\tempty\n";
			++emptied;
			continue;
		}

		bool const isDna{ looks_like_dna(text) };
		if (isDna && !opt.dna)
		{
			cout << '\t' << sizeIn << "\tgenome, ignored\n";
			++dnaIgnored;
			continue;
		}
		charsIn += sizeIn;

		string note;
		size_t headCut{ 0 }, tailCut{ 0 }, comboRemoved{ 0 };
		ComboStats comboFile;
		if (isDna)
		{
			text = extract_genome(text);
			note = "genome";
		}
		else
		{
			if (opt.autoCut)
			{
				auto const cut{ boiler.find_cut(text) };
				headCut = cut.head;
				tailCut = text.size() - cut.tail;
				text.erase(cut.tail);
				text.erase(0, cut.head);
				autoHeadFiles += headCut > 0;
				autoTailFiles += tailCut > 0;
				autoHeadChars += headCut;
				autoTailChars += tailCut;
			}
			if (opt.gutten && !strip_gutenberg(text))
			{
				++noMarkers;
				note = headCut || tailCut ? "no Gutenberg markers" : "no Gutenberg markers, kept as is";
			}
			if (opt.norm)
				text = normalize_whitespace(text);
			if (opt.combo)
				comboRemoved = collapse_runs(text, opt.comboMax, &comboFile);
		}

		cout << '\t' << text.size() << " (" << delta_str(sizeIn, text.size()) << ")";
		if (opt.autoCut && !isDna)
			cout << "\thead -" << headCut << ", tail -" << tailCut;

		if (opt.combo && !isDna)
			cout << "\tcombo -" << comboRemoved;

		double const digits{ isDna ? 0.0 : digit_share(text) };
		if (!isDna)
			cout << '\t' << pct_str(digits) << " digits";
		if (!note.empty())
			cout << '\t' << note;

		bool const tooManyDigits{ opt.maxDigits > 0 && digits > opt.maxDigits };
		if (tooManyDigits)
			cout << "\tSKIPPED: too many digits";
		cout << '\n';

		if (tooManyDigits)
		{
			charsIn -= sizeIn;   // not part of the corpus, so not a "loss" either
			digitSkippedChars += sizeIn;
			++digitSkipped;
			digitList.emplace_back(f.filename(), digits);
			continue;
		}

		// The file is part of the corpus now: its combo statistics count.
		if (comboRemoved > 0)
		{
			++comboFiles;
			comboChars += comboRemoved;
			for (auto const& [sym, st] : comboFile)
			{
				auto& tot{ comboTotal[sym] };
				tot.runs += st.runs;
				tot.removed += st.removed;
				tot.longest = max(tot.longest, st.longest);
			}
		}

		if (text.empty())
		{
			++emptied;
			continue;
		}

		// Files must not fuse: end every file with a blank line.
		if (!text.ends_with("\n\n"))
			text += text.ends_with('\n') ? "\n" : "\n\n";

		if (opt.big)
			out.write(text.data(), static_cast<streamsize>(text.size()));
		charsOut += text.size();
		++used;
	}

	// ------------------------------------------------------------------
	// Summary
	// ------------------------------------------------------------------
	cout << "\n=== Summary ===\n"
		<< "Files found      : " << files.size() << "\n"
		<< "Files used       : " << used << "\n"
		<< "Genome ignored   : " << dnaIgnored << "\n"
		<< "Failed to read   : " << failed << "\n"
		<< "Empty after prep : " << emptied << "\n";
	if (opt.gutten)
		cout << "No markers       : " << noMarkers << "\n";
	if (skippedOld)
		cout << "Skipped input    : previous corpus.txt\n";
	if (opt.maxDigits > 0)
		cout << "Skipped (digits) : " << digitSkipped << " files, " << digitSkippedChars << " chars\n";
	if (opt.autoCut)
		cout << "Auto cut head    : " << autoHeadFiles << " files, " << autoHeadChars << " chars\n"
		<< "Auto cut tail    : " << autoTailFiles << " files, " << autoTailChars << " chars\n";
	if (opt.combo)
		cout << "Combo cut        : " << comboChars << " symbols in " << comboFiles << " files (runs longer than " << opt.comboMax << ")\n";
	cout << "Characters in    : " << charsIn << "\n"
		<< "Characters out   : " << charsOut << " (" << delta_str(charsIn, charsOut) << ", file separators included)\n";

	if (opt.combo)
	{
		print_combo_report(comboTotal, opt.comboMax);
	}

	for (auto const& [name, pct] : digitList)
		cout << "\tskipped " << name << "\t" << pct_str(pct) << " digits\n";

	if (!opt.big)
	{
		cout << "Nothing saved (use /big to write corpus.txt)\n" << endl;
		return EXIT_SUCCESS;
	}

	out.close();
	if (!out)
	{
		cerr << "Error: writing " << corpusPath << " failed.\n";
		return EXIT_FAILURE;
	}
	cout << "Corpus saved to  : " << corpusPath << "\n" << endl;
	return EXIT_SUCCESS;
}
