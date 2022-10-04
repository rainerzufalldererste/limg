ProjectName = "limg"
project(ProjectName)

  --Settings
  kind "ConsoleApp"
  language "C++"
  flags { "FatalWarnings" }
  staticruntime "On"

  filter {"system:windows"}
    buildoptions { '/MP' }
    ignoredefaultlibraries { "msvcrt" }

  filter { }
    cppdialect "C++17"
  
  filter { }
  
  defines { "_CRT_SECURE_NO_WARNINGS" }
  
  objdir "intermediate/obj"

  files { "src/**.c", "src/**.cc", "src/**.cpp", "src/**.h", "src/**.hh", "src/**.hpp", "src/**.inl", "src/**rc", "*.md" }
  files { "project.lua" }
  
  includedirs { "src**" }
  includedirs { "3rdParty/stb/include" }
  
  targetname(ProjectName)
  targetdir "builds/bin"
  debugdir "builds/bin"
  
filter {}

filter {}
warnings "Extra"

filter { }
  exceptionhandling "Off"
  rtti "Off"
  floatingpoint "Fast"

filter { "configurations:Debug*" }
	defines { "_DEBUG" }
	optimize "Off"
	symbols "FastLink"

filter { "configurations:Release*" }
	defines { "NDEBUG" }
	optimize "Speed"
	flags { "NoBufferSecurityCheck" }
  omitframepointer "On"
  symbols "On"

filter { "system:windows" }
	defines { "WIN32", "_WINDOWS" }
  flags { "NoPCH", "NoMinimalRebuild" }
	links { "kernel32.lib", "user32.lib", "gdi32.lib", "winspool.lib", "comdlg32.lib", "advapi32.lib", "shell32.lib", "ole32.lib", "oleaut32.lib", "uuid.lib", "odbc32.lib", "odbccp32.lib" }
  
filter { "system:windows", "configurations:Release" }
  flags { "NoIncrementalLink" }

filter { "system:windows", "configurations:Debug" }
  ignoredefaultlibraries { "libcmt" }
filter { }