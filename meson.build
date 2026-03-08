project('vs-cfl', 'cpp',
    version: '1.0.0',
    default_options: [
        'cpp_std=c++latest', # c++23
        'buildtype=release',
        'b_lto=true',
    ])

sources = files(
    'src/vs_cfl.cpp',
    'src/kACfL.cpp',
)

shared_library('vs_cfl', sources,
    include_directories: include_directories('include', 'src'),
    name_prefix: '',
    gnu_symbol_visibility: 'hidden',
    install: true)
