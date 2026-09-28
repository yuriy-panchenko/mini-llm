// GuttenbergLibrary.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include "..\mini_llm_core\tokenizer.h"
#include "..\mini_llm_core\classmap.h"

//constexpr auto source_root{ "..\\D184MB\\" };
//constexpr auto source_root{ "..\\D1GB\\" };
//constexpr auto source_root{ "..\\D1.7GB\\" };
//constexpr auto source_root{ "..\\D1GB\\Genome\\" };
//constexpr auto destin_root{ "..\\Vocabs\\" };

//constexpr size_t vocab_size{ 0x8000 };
constexpr size_t max_vocab_size{ 0xFFF0 };

using namespace std;
using namespace llm;
using namespace chrono;
namespace fs = filesystem;
//namespace clk = chrono;

Tokenizer gTok;

std::string GetGenome(std::ifstream& s)
{
	std::string const src{ istreambuf_iterator<char>{s}, istreambuf_iterator<char>{} };
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

void save_as(fs::path fn)
{
	cout << "\n\nSaving dictionary " << fn.filename() << "\t............\t";
	std::ofstream file{ fn, ios::binary };
	if (file)
	{
		file << gTok;
		cout << "OK!";
	}
	else cout << "ERR";

	cout << endl;
}


// Replaces every occurrence of `from` with `to`, in place.
// Handles overlapping/empty-replacement cases safely by advancing past each replacement.
// Usage:
// replace_all(text, "Bot: ", "");      // remove
// replace_all(text, "Bot: ", "A: ");   // replace
void replace_all(std::string& s, const std::string& from, const std::string& to)
{
	if (from.empty()) return;

	size_t pos = 0;
	while ((pos = s.find(from, pos)) != std::string::npos)
	{
		s.replace(pos, from.size(), to);
		pos += to.size();   // skip past what we just inserted
	}
}

std::string replace_all_fast(const std::string& s, const std::string& from, const std::string& to)
{
	if (from.empty()) return s;

	std::string out;
	out.reserve(s.size());

	size_t pos = 0, hit;
	while ((hit = s.find(from, pos)) != std::string::npos)
	{
		out.append(s, pos, hit - pos);   // copy the untouched stretch
		out += to;
		pos = hit + from.size();
	}
	out.append(s, pos, std::string::npos);  // tail
	return out;
}

int main(int argc, char const* argv[])
{
	// Default values
	fs::path src_path;
	fs::path dst_path = fs::current_path();   // current by default
	bool do_norm = false;
	bool do_dna = false;
	bool do_big = false;
	size_t vocab_size = max_vocab_size;       // default max

	// ------------------------------------------------------------------
	// Command-line parsing
	// ------------------------------------------------------------------
	if (argc == 1)
	{
		cout << "gutlib src [dst] [flags]\n"
			<< "\tsrc\t\t- path to file or folder with .txt files\n"
			<< "\tdst\t\t- destination folder path (current by default)\n"
			<< "\t/norm\t\t- normalize whitespaces\n"
			<< "\t/dna\t\t- open file as Genome file\n"
			<< "\t/big\t\t- save combined and cleaned corpus to corpus.txt file\n"
			<< "\t/size=1024\t- max vocabulary size\n"
			<< endl;
		return EXIT_SUCCESS;
	}

	// Collect positional arguments and flags
	vector<string> positional;
	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];

		// Flags (case-insensitive comparison for convenience)
		if (arg == "/norm" || arg == "/NORM" || arg == "-norm")
		{
			do_norm = true;
		}
		else if (arg == "/dna" || arg == "/DNA" || arg == "-dna")
		{
			do_dna = true;
		}
		else if (arg == "/big" || arg == "/BIG" || arg == "-big")
		{
			do_big = true;
		}
		else if (arg.rfind("/size=", 0) == 0 || arg.rfind("/SIZE=", 0) == 0 || arg.rfind("-size=", 0) == 0)
		{
			// Parse /size=N or -size=N
			string value = arg.substr(arg.find('=') + 1);
			try
			{
				size_t parsed = stoull(value);
				if (parsed == 0 || parsed > max_vocab_size)
				{
					cerr << "Error: vocabulary size must be between 1 and " << max_vocab_size << endl;
					return EXIT_FAILURE;
				}
				vocab_size = parsed;
			}
			catch (...)
			{
				cerr << "Error: invalid vocabulary size value: " << value << endl;
				return EXIT_FAILURE;
			}
		}
		else if (arg[0] == '/' || arg[0] == '-')
		{
			cerr << "Unknown flag: " << arg << endl;
			return EXIT_FAILURE;
		}
		else
		{
			// Positional argument
			positional.push_back(arg);
		}
	}

	// Assign positional arguments
	if (positional.empty())
	{
		cerr << "Error: source path (src) is required.\n";
		return EXIT_FAILURE;
	}

	src_path = positional[0];
	if (positional.size() >= 2)
		dst_path = positional[1];

	// Validate paths
	if (!fs::exists(src_path))
	{
		cerr << "Error: source path does not exist: " << src_path << endl;
		return EXIT_FAILURE;
	}

	if (!fs::exists(dst_path))
	{
		cerr << "Destination folder does not exist. Creating: " << dst_path << endl;
		fs::create_directories(dst_path);
	}

	// ------------------------------------------------------------------
	// Info display
	// ------------------------------------------------------------------
	cout << "=== Gutenberg Library Tokenizer ===\n"
		<< "Source      : " << src_path << "\n"
		<< "Destination : " << dst_path << "\n"
		<< "Normalize   : " << (do_norm ? "yes" : "no") << "\n"
		<< "DNA mode    : " << (do_dna ? "yes" : "no") << "\n"
		<< "Save corpus : " << (do_big ? "yes" : "no") << "\n"
		<< "Vocab size  : " << vocab_size << " (0x" << hex << vocab_size << dec << ")\n"
		<< "===================================\n" << endl;

	// ------------------------------------------------------------------
	// Load corpus
	// ------------------------------------------------------------------
	std::string corpus;

	if (do_dna)
	{
		// Treat as single Genome file
		if (!fs::is_regular_file(src_path))
		{
			cerr << "Error: /dna expects a single file, not a directory.\n";
			return EXIT_FAILURE;
		}
		std::ifstream fs{ src_path };
		if (!fs)
		{
			cerr << "Error: cannot open genome file: " << src_path << endl;
			return EXIT_FAILURE;
		}
		corpus = GetGenome(fs);
		cout << "Genome sequence length: " << corpus.size() << " bases\n";
	}
	else
	{
		// Normal text mode: file or folder of .txt files
		if (fs::is_directory(src_path))
		{
			corpus = llm::read_all_files(src_path.string());
		}
		else
		{
			corpus = llm::read_file(src_path.string());
		}
	}

	if (do_norm)
	{
		corpus = llm::normalize_whitespace(corpus);
		cout << "Whitespace normalized.\n";
	}

	if (do_big)
	{
		fs::path corpus_file = dst_path / "corpus.txt";
		std::ofstream out{ corpus_file };
		if (out)
		{
			out << corpus;
			cout << "Combined corpus saved to: " << corpus_file << "\n";
		}
		else
		{
			cerr << "Warning: could not write corpus.txt\n";
		}
	}

	// ------------------------------------------------------------------
	// Training loop (unchanged)
	// ------------------------------------------------------------------
	gTok.reset(vocab_size);
	gTok.tokenize(corpus);

	size_t milestone{ 0x200ull }, max_len{ 1ull }, best_count;
	std::string cmb_text;
	bool just_saved{ false };

	while (gTok.vocab_size() < vocab_size)
	{
		just_saved = false;
		auto const tpStart{ steady_clock::now() };
		auto const best{ gTok.find_most_used_pair(&best_count) };
		if (best == Tokenizer::invalid_pair)
			break;
		auto tok_size{ gTok.get_tokens().size() };
		gTok.unite(best, best_count);
		auto const tpEnd{ steady_clock::now() };

		cmb_text = gTok.text(best);
		max_len = std::max(max_len, cmb_text.length());
		std::cout
			<< '\n' << tok_size
			<< ", lib " << gTok.vocab_size()
			<< ", max " << max_len
			<< '\t' << fixed << setprecision(3) << duration_cast<milliseconds>(tpEnd - tpStart).count() / 1000.
			<< '\t' << best_count << "\t\"" << cmb_text << '\"'
			;

		if (gTok.vocab_size() == milestone)
		{
			fs::path save_path = dst_path / ("token" + (milestone >= 0x400 ? to_string(milestone / 0x400) + "K" : to_string(milestone)) + ".bin");
			save_as(save_path);
			milestone <<= 1;
			just_saved = true;
		}
	}

	if (!just_saved)
		save_as(dst_path / "tokens.bin");
}