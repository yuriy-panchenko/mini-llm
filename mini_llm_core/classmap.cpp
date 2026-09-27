#include "pch.h"
#include "classmap.h"
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <fstream>

namespace llm
{
	using namespace std;
	namespace fs = std::filesystem;

	std::string read_all_files(filesystem::path const& root)
	{
		string ret;

		cout << "Reading \'*.txt\' files in " << root << "\n\n";

		size_t file_count{};
		for (auto const& en : fs::directory_iterator{ root })
			if (en.is_regular_file() && en.path().extension() == ".txt")
			{
				string text;
				try
				{
					cout << file_count + 1 << " " << en.path().filename();
					text = read_file(en.path());
					++file_count;
					cout << '\t' << fs::file_size(en.path());

					if (file_count % 4)
						cout << '\t';
					else cout << '\n';
				}
				catch (const std::exception&)
				{
					cout << "\tFailed";
				}
				ret.append(std::move(text));
			}

		cout << "\n===============================================================================================\
		\nAll Files\t" << file_count
			<< "\nCharacters\t" << ret.length() << endl;

		return ret;
	}
	
	std::string read_file(fs::path const& filename)
	{
		std::ifstream file{ filename, std::ios::binary };
		if (file)
			return { istreambuf_iterator<char>{file},istreambuf_iterator<char>{} };
		else throw std::runtime_error("Cannot open file: " + filename.string());
	}
}