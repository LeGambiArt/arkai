%global srcname arkai
%global _description %{expand:
Arkai is a command-line tool for orchestrating coding agents with local or
cloud inference. It can manage model downloads, agent sessions, MCP
integrations, and sandboxed execution.}

Name:           python3-arkai
Version:        0.10.0
Release:        %autorelease
Summary:        Local AI workflow manager for coding agents
License:        Apache-2.0
URL:            https://github.com/LeGambiArt/arkai
Source0:        https://github.com/LeGambiArt/arkai/archive/refs/tags/v%{version}.tar.gz
BuildArch:      noarch

BuildRequires:  python3-devel
BuildRequires:  python3dist(behave)
BuildRequires:  python3dist(pytest)

# Arkai's keychain integration is installed with the main package.
Requires:       python3dist(keyring) >= 24.0

%description
%{_description}

# Provide the upstream RAG extra as a dependency-only Fedora subpackage.
%pyproject_extras_subpkg -n %{name} rag

%prep
%autosetup -p1 -n %{srcname}-%{version}

%generate_buildrequires
%pyproject_buildrequires

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files -l arkai

%check
%pytest tests/unit
%{python3} -m behave features

%files -n %{name} -f %{pyproject_files}
%doc README.md
%{_bindir}/arkai

%changelog
%autochangelog
