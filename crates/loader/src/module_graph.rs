// SPDX-License-Identifier: GPL-3.0-or-later

//! A bounded, non-executing dependency graph for explicitly supplied PE32+
//! images.
//!
//! This module is loader infrastructure only. It reads ordinary import and
//! export directories from bytes the caller already supplied, resolves those
//! references within that explicit package and reports a deterministic
//! dependency order. It does not search a host filesystem, load or map images,
//! resolve API sets or runtime services, write import address tables, run
//! relocation, TLS or initialization code, count references, unload modules or
//! execute guest instructions.
//!
//! Package input names are logical bare file names, not paths. Xenora accepts
//! only nonempty printable ASCII names of at most [`MAX_PACKAGE_FILE_NAME_LEN`]
//! bytes, rejects `/` and `\\`, and rejects a trailing dot because a trailing
//! dot has reference-normalization meaning rather than being part of the
//! no-extension name. Canonical package identity is ASCII case-insensitive.
//! This input identity is deliberately independent of the DLL name string
//! stored inside an image's export directory; the latter is metadata in the
//! image and is never used to look up a package module.
//!
//! Module references in import descriptors and forwarder strings use the
//! documented Windows spelling rules: comparison is ASCII case-insensitive, a
//! reference with no dot is treated as if it ended in `.dll`, and one trailing
//! dot suppresses that appending and is removed, so `library.` refers to the
//! no-extension module `library`. Exported symbol names are case-sensitive.
//! References containing `/`, `\` or `:` are unsupported rather than search
//! paths. API-set contract references beginning `api-` or `ext-` are diagnosed
//! as outside this package graph even when a same-named file was supplied.
//!
//! PE forwarder strings do not have a fully documented public grammar for DLL
//! names containing dots. Xenora uses the compatibility policy observed in
//! Windows-compatible loaders and splits at the last period: the left part is
//! the DLL reference (normalized by the rules above) and the right part is the
//! case-sensitive symbol or `#` followed by a decimal ordinal. Forwarding is
//! iterative, bounded to [`MAX_FORWARDER_HOPS`] hops, and never treats the
//! forwarder string or its target as code.
//!
//! Import and export parsers can retain entries read before the first table
//! error. Those structured errors are preserved on each module node. A
//! malformed export table is never used for resolution, even if its partial
//! entries contain a matching name. Header bytes are valid backing, but bytes
//! present only through section virtual zero-fill are not yet represented as
//! readable table bytes. [`ModuleGraph::inspection_is_complete`] tells an
//! executing consumer whether table inspection completed; it does not mean
//! that every import resolved.
//!
//! Dependency edges describe references needed by observed import resolution.
//! They include every ordinary import's existing provider module. When a
//! resolution reaches a forwarder, they also include an edge from the module
//! containing that forwarder to the forwarder's in-package target. An unused
//! forwarder therefore does not create a dependency edge or a circular
//! component. This is static binding evidence, not a claimed Windows
//! initialization order.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::export_table::{ExportTable, ExportTableError, ExportTarget, parse_exports};
use crate::import_table::{ImportEntry, ImportName, ImportTableError, parse_imports};
use crate::pe::{ParsedPe, PeError};

/// Most modules accepted in one explicit package. This Xenora policy bound
/// keeps graph construction and cycle analysis finite for hostile input.
pub const MAX_PACKAGE_MODULES: usize = 256;
/// Longest logical package file name accepted, in bytes. This is a Xenora
/// policy bound, not a PE format limit.
pub const MAX_PACKAGE_FILE_NAME_LEN: usize = 255;
/// Most forwarder edges followed while resolving one imported symbol. This is
/// a Xenora policy bound; cycles are also diagnosed when detected earlier.
pub const MAX_FORWARDER_HOPS: usize = 8;

const EXPORT_DIRECTORY: usize = 0;
const IMPORT_DIRECTORY: usize = 1;
const OPTIONAL_FIXED_LEN: usize = 112;
const DIRECTORY_ENTRY_LEN: usize = 8;

/// One caller-supplied in-memory image. `file_name` is the logical package
/// identity described in the module documentation; `bytes` are the complete
/// PE32+ image bytes.
#[derive(Clone, Copy, Debug)]
pub struct PackageModule<'a> {
    pub file_name: &'a str,
    pub bytes: &'a [u8],
}

/// A module identified by its stable package index and logical file name.
/// Package indexes are the order in which modules were supplied to
/// [`ModuleGraph::build`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModuleReference {
    pub index: usize,
    pub file_name: String,
}

impl fmt::Display for ModuleReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (package index {})", self.file_name, self.index)
    }
}

/// Why a logical package file name was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileNameError {
    Empty,
    TooLong {
        len: usize,
        max: usize,
    },
    NotPrintableAscii,
    PathSeparator,
    /// A colon is Windows drive/stream punctuation, not part of a bare module
    /// file name.
    DriveSeparator,
    /// A trailing dot is reference punctuation for a no-extension name. The
    /// no-extension module itself must be supplied without the dot.
    TrailingDot,
}

impl fmt::Display for FileNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("file name is empty"),
            Self::TooLong { len, max } => {
                write!(f, "file name length {len} exceeds the {max}-byte limit")
            }
            Self::NotPrintableAscii => f.write_str("file name is not bounded printable ASCII"),
            Self::PathSeparator => f.write_str("file name contains a path separator"),
            Self::DriveSeparator => f.write_str("file name contains a drive separator"),
            Self::TrailingDot => f.write_str("file name ends in a dot"),
        }
    }
}

/// A PE32+ optional-header data directory read by this graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataDirectory {
    Export,
    Import,
}

impl fmt::Display for DataDirectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Export => "export",
            Self::Import => "import",
        })
    }
}

/// Input and PE structure errors that prevent a package graph from being
/// constructed. Import/export table contents have separate partial errors on
/// [`ModuleNode`].
#[derive(Debug, PartialEq, Eq)]
pub enum ModuleGraphError {
    TooManyModules {
        count: usize,
        max: usize,
    },
    InvalidFileName {
        index: usize,
        file_name: String,
        reason: FileNameError,
    },
    DuplicateModuleIdentity {
        first_index: usize,
        second_index: usize,
        /// ASCII-lowercase canonical identity shared by the two inputs.
        identity: String,
    },
    InvalidPe {
        module: ModuleReference,
        error: PeError,
    },
    /// `NumberOfRvaAndSizes` admits this directory, but the declared optional
    /// header does not contain the whole entry. This distinction is required
    /// by the PE format's data-directory probing rule.
    TruncatedDataDirectory {
        module: ModuleReference,
        directory: DataDirectory,
    },
}

impl fmt::Display for ModuleGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyModules { count, max } => {
                write!(
                    f,
                    "package has {count} modules, exceeding the {max}-module limit"
                )
            }
            Self::InvalidFileName {
                index,
                file_name,
                reason,
            } => write!(
                f,
                "module {index} file name {file_name:?} is invalid: {reason}"
            ),
            Self::DuplicateModuleIdentity {
                first_index,
                second_index,
                identity,
            } => write!(
                f,
                "modules {first_index} and {second_index} both use canonical identity {identity:?}"
            ),
            Self::InvalidPe { module, error } => {
                write!(f, "module {module} is not a supported PE32+ image: {error}")
            }
            Self::TruncatedDataDirectory { module, directory } => write!(
                f,
                "module {module} truncates its {directory} data-directory entry"
            ),
        }
    }
}

impl std::error::Error for ModuleGraphError {}

/// Static import/export inspection data for one package module. Entries and
/// errors are exactly the bounded partial results of the import and export
/// parsers. A present error means the corresponding table is incomplete.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleNode {
    /// Logical input file name used for package identity.
    pub file_name: String,
    /// ASCII-lowercase form of `file_name`, used for module lookup.
    pub canonical_file_name: String,
    pub is_dll: bool,
    /// Ordinary imports read before the first import-table error, or all
    /// ordinary imports when `import_error` is absent. Delayed imports are
    /// outside this graph's scope.
    pub imports: Vec<ImportEntry>,
    pub import_error: Option<ImportTableError>,
    /// Export table retained for inspection. Its embedded DLL name is image
    /// metadata, not the package identity used for lookup.
    pub exports: Option<ExportTable>,
    pub export_error: Option<ExportTableError>,
}

/// A dependency edge from the consuming module to the providing module. The
/// edge exists whenever the referenced provider exists in the package, even if
/// the particular imported or forwarded symbol does not resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DependencyEdge {
    pub consumer: usize,
    pub provider: usize,
}

/// One item in deterministic provider-before-consumer dependency order. A
/// circular component is explicit and is never represented as a valid
/// initialization order for its members.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DependencyComponent {
    /// One module with no dependency cycle. The value is its package index.
    Acyclic(usize),
    /// Package indexes of one strongly connected circular component, sorted
    /// ascending for determinism. This order is only an identity order, not an
    /// initialization order. A self-dependent module appears here even when it
    /// is the only member.
    Circular(Vec<usize>),
}

/// How an export is selected. Unlike import ordinals, export ordinals are
/// represented as u32 because the PE export ordinal base is a u32 field.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExportKey {
    Name(String),
    Ordinal(u32),
}

impl fmt::Display for ExportKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(name) => write!(f, "name {name:?}"),
            Self::Ordinal(ordinal) => write!(f, "ordinal #{ordinal}"),
        }
    }
}

/// Why a forwarder string is malformed or unsupported under this graph's
/// documented last-period grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwarderParseError {
    MissingSeparator,
    EmptyModuleName,
    EmptySymbolName,
    EmptyOrdinal,
    InvalidOrdinalDigit,
    OrdinalOverflow,
    /// The DLL component contains `/`, `\` or `:`. This graph never treats
    /// such text as a host or package search path.
    ModulePathSeparator,
}

impl fmt::Display for ForwarderParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MissingSeparator => "forwarder has no DLL/symbol separator",
            Self::EmptyModuleName => "forwarder DLL name is empty",
            Self::EmptySymbolName => "forwarder symbol name is empty",
            Self::EmptyOrdinal => "forwarder ordinal is empty",
            Self::InvalidOrdinalDigit => "forwarder ordinal is not decimal ASCII",
            Self::OrdinalOverflow => "forwarder ordinal does not fit in u32",
            Self::ModulePathSeparator => "forwarder DLL name contains a path or drive separator",
        })
    }
}

impl std::error::Error for ForwarderParseError {}

/// One parsed forwarder edge encountered during import resolution.
/// `provider` and `export` identify the export that forwards; `target_module`
/// is normalized, and `target_export` is the symbol requested from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForwarderStep {
    pub provider: ModuleReference,
    pub export: ExportKey,
    pub target: String,
    pub target_module: String,
    pub target_export: ExportKey,
}

/// Why an import did not resolve. In a forwarded resolution, provider/export
/// data in the reason or its terminal step describes the point at which
/// resolution failed; the import's `forwarder_chain` records the path already
/// traversed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnresolvedImportReason {
    /// The normalized module reference is absent from the explicit package.
    MissingModule { module: String },
    /// The import uses `/`, `\` or `:` in its module reference. Xenora does
    /// not turn module references into host or package search paths.
    UnsupportedModulePath { module: String },
    /// The import names an API-set contract rather than an ordinary module.
    /// API-set schema resolution is explicitly outside this graph, and a
    /// same-named package file never satisfies the contract.
    ApiSetContract { contract: String },
    /// A parsed forwarder names an API-set contract. The step identifies the
    /// forwarding provider and export.
    ForwarderApiSetContract { step: ForwarderStep },
    /// A parsed forwarder names a module absent from the explicit package.
    /// The step identifies the forwarding provider and export as well as the
    /// missing normalized target.
    ForwarderMissingModule { step: ForwarderStep },
    /// The provider's export table has a structural error. Partial entries are
    /// deliberately not used for resolution; the structured error is preserved
    /// on the provider's [`ModuleNode`].
    ProviderExportsMalformed { provider: ModuleReference },
    /// The provider's export inspection is complete, but it has no entry for
    /// this exact case-sensitive name or biased ordinal. The provider may have
    /// no export directory at all.
    MissingExport {
        provider: ModuleReference,
        export: ExportKey,
    },
    /// The requested key selects an export address-table slot whose RVA is
    /// zero. Such a slot is an absent export, not code at the image base.
    EmptyExportSlot {
        provider: ModuleReference,
        export: ExportKey,
        /// Biased ordinal of the empty address-table slot.
        ordinal: u32,
    },
    /// The forwarder text is printable inventory data but does not satisfy
    /// this graph's documented DLL/symbol grammar.
    MalformedForwarder {
        provider: ModuleReference,
        export: ExportKey,
        target: String,
        error: ForwarderParseError,
    },
    /// The parsed forwarder in `step` would return to a module/export pair
    /// already present in the current resolution chain.
    ForwarderCycle { step: ForwarderStep },
    /// The parsed forwarder in `step` would exceed [`MAX_FORWARDER_HOPS`].
    ForwarderHopLimit { step: ForwarderStep },
}

/// The static resolution outcome for one imported symbol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportOutcome {
    /// The final in-package direct export. For a forwarded import this is the
    /// provider and export reached after the recorded forwarder chain. The RVA
    /// is only inventory data: nothing has been mapped, relocated or executed.
    Resolved {
        provider: ModuleReference,
        export: ExportKey,
        /// Biased ordinal of the final export address-table slot.
        ordinal: u32,
        rva: u32,
    },
    Unresolved(UnresolvedImportReason),
}

/// Deterministic resolution information for one ordinary import. `consumer`,
/// `requested_module` and `symbol` identify the import; `normalized_module` is
/// the lookup spelling after Windows module-reference normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportResolution {
    pub consumer: ModuleReference,
    pub requested_module: String,
    pub normalized_module: String,
    pub symbol: ImportName,
    pub iat_rva: u32,
    /// Forwarders successfully parsed and traversed, in order. A terminal
    /// forwarder that cannot be traversed is instead inside the unresolved
    /// reason when the reason needs its precise source and target.
    pub forwarder_chain: Vec<ForwarderStep>,
    pub outcome: ImportOutcome,
}

/// A deterministic static package graph. Use [`ModuleGraph::build`] to
/// construct it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleGraph {
    modules: Vec<ModuleNode>,
    dependency_edges: Vec<DependencyEdge>,
    dependency_order: Vec<DependencyComponent>,
    import_resolutions: Vec<ImportResolution>,
}

impl ModuleGraph {
    /// Parse and connect an explicit package of in-memory PE32+ images.
    ///
    /// Validation of logical names and duplicate canonical identities happens
    /// before any image parsing. Image parsing and graph construction are
    /// deterministic; no bytes are loaded as executable images.
    pub fn build(package: &[PackageModule<'_>]) -> Result<Self, ModuleGraphError> {
        if package.len() > MAX_PACKAGE_MODULES {
            return Err(ModuleGraphError::TooManyModules {
                count: package.len(),
                max: MAX_PACKAGE_MODULES,
            });
        }

        let mut package_modules = BTreeMap::new();
        for (index, input) in package.iter().enumerate() {
            if let Err(reason) = validate_file_name(input.file_name) {
                return Err(ModuleGraphError::InvalidFileName {
                    index,
                    file_name: input.file_name.to_owned(),
                    reason,
                });
            }
            let canonical = input.file_name.to_ascii_lowercase();
            if let Some(first_index) = package_modules.insert(canonical.clone(), index) {
                return Err(ModuleGraphError::DuplicateModuleIdentity {
                    first_index,
                    second_index: index,
                    identity: canonical,
                });
            }
        }

        let mut modules = Vec::with_capacity(package.len());
        for (index, input) in package.iter().enumerate() {
            modules.push(parse_module(index, input)?);
        }

        let export_indexes: Vec<_> = modules.iter().map(export_index).collect();
        let mut edge_pairs = BTreeSet::new();
        let mut import_resolutions = Vec::new();
        for (consumer, node) in modules.iter().enumerate() {
            for import in &node.imports {
                let normalized = normalize_module_reference(&import.module);
                if !has_module_path_separator(&import.module)
                    && !is_api_set_reference(&import.module)
                    && let Some(&provider) = package_modules.get(&normalized)
                {
                    edge_pairs.insert((consumer, provider));
                }
                let resolution = resolve_import(
                    &modules,
                    &export_indexes,
                    &package_modules,
                    consumer,
                    import,
                );
                add_resolution_forwarder_edges(&resolution, &package_modules, &mut edge_pairs);
                import_resolutions.push(resolution);
            }
        }

        let adjacency = adjacency_lists(modules.len(), &edge_pairs);
        let components = strongly_connected_components(modules.len(), &adjacency);
        let dependency_order = order_components(modules.len(), &components, &adjacency);
        let dependency_edges = edge_pairs
            .into_iter()
            .map(|(consumer, provider)| DependencyEdge { consumer, provider })
            .collect();

        Ok(Self {
            modules,
            dependency_edges,
            dependency_order,
            import_resolutions,
        })
    }

    /// Module nodes in input package order.
    pub fn modules(&self) -> &[ModuleNode] {
        &self.modules
    }

    /// Deduplicated consumer-to-provider edges, sorted by package indexes.
    pub fn dependency_edges(&self) -> &[DependencyEdge] {
        &self.dependency_edges
    }

    /// Deterministic provider-before-consumer condensation order. Circular
    /// strongly connected components remain grouped and explicitly marked.
    /// When several components are ready together, the one whose smallest
    /// package index is lowest is emitted first.
    pub fn dependency_order(&self) -> &[DependencyComponent] {
        &self.dependency_order
    }

    /// Import resolutions in input module order, preserving each module's
    /// ordinary import-table order.
    pub fn import_resolutions(&self) -> &[ImportResolution] {
        &self.import_resolutions
    }

    /// Whether every ordinary import table and export directory parsed without
    /// a structural error. This only describes inspection completeness; it does
    /// not imply that referenced modules or exports exist.
    pub fn inspection_is_complete(&self) -> bool {
        self.modules
            .iter()
            .all(|node| node.import_error.is_none() && node.export_error.is_none())
    }
}

fn validate_file_name(file_name: &str) -> Result<(), FileNameError> {
    if file_name.is_empty() {
        return Err(FileNameError::Empty);
    }
    if file_name.len() > MAX_PACKAGE_FILE_NAME_LEN {
        return Err(FileNameError::TooLong {
            len: file_name.len(),
            max: MAX_PACKAGE_FILE_NAME_LEN,
        });
    }
    if !file_name
        .bytes()
        .all(|byte| byte.is_ascii_graphic() || byte == b' ')
    {
        return Err(FileNameError::NotPrintableAscii);
    }
    if file_name.bytes().any(|byte| matches!(byte, b'/' | b'\\')) {
        return Err(FileNameError::PathSeparator);
    }
    if file_name.bytes().any(|byte| byte == b':') {
        return Err(FileNameError::DriveSeparator);
    }
    if file_name.ends_with('.') {
        return Err(FileNameError::TrailingDot);
    }
    Ok(())
}

/// Normalize a Windows module reference for this explicit package. Only ASCII
/// case is folded. Extension handling is part of the reference, not evidence
/// that a host file exists.
fn normalize_module_reference(reference: &str) -> String {
    let mut normalized = reference.to_ascii_lowercase();
    if normalized.ends_with('.') {
        normalized.pop();
    } else if !normalized.contains('.') {
        normalized.push_str(".dll");
    }
    normalized
}

fn has_module_path_separator(reference: &str) -> bool {
    reference
        .bytes()
        .any(|byte| matches!(byte, b'/' | b'\\' | b':'))
}

fn is_api_set_reference(reference: &str) -> bool {
    let lowercase = reference.to_ascii_lowercase();
    lowercase.starts_with("api-") || lowercase.starts_with("ext-")
}

fn module_reference(modules: &[ModuleNode], index: usize) -> ModuleReference {
    ModuleReference {
        index,
        file_name: modules[index].file_name.clone(),
    }
}

fn parse_module(index: usize, input: &PackageModule<'_>) -> Result<ModuleNode, ModuleGraphError> {
    let module = ModuleReference {
        index,
        file_name: input.file_name.to_owned(),
    };
    let pe = ParsedPe::parse(input.bytes).map_err(|error| ModuleGraphError::InvalidPe {
        module: module.clone(),
        error,
    })?;
    let export_directory = data_directory(input.bytes, EXPORT_DIRECTORY).map_err(|()| {
        ModuleGraphError::TruncatedDataDirectory {
            module: module.clone(),
            directory: DataDirectory::Export,
        }
    })?;
    let import_directory = data_directory(input.bytes, IMPORT_DIRECTORY).map_err(|()| {
        ModuleGraphError::TruncatedDataDirectory {
            module: module.clone(),
            directory: DataDirectory::Import,
        }
    })?;

    // A zero data-directory RVA means that the table is absent; the size field
    // is not interpreted as a table at the image headers.
    let (exports, export_error) = if export_directory.0 == 0 {
        (None, None)
    } else {
        parse_exports(input.bytes, export_directory.0, export_directory.1, |rva| {
            backed_file_offset(&pe, input.bytes, rva)
        })
    };
    let (imports, import_error) = if import_directory.0 == 0 {
        (Vec::new(), None)
    } else {
        parse_imports(input.bytes, import_directory.0, import_directory.1, |rva| {
            backed_file_offset(&pe, input.bytes, rva)
        })
    };

    Ok(ModuleNode {
        file_name: input.file_name.to_owned(),
        canonical_file_name: input.file_name.to_ascii_lowercase(),
        is_dll: pe.is_dll(),
        imports,
        import_error,
        exports,
        export_error,
    })
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(at..at.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

/// Read a PE32+ data directory without probing beyond the declared optional
/// header or `NumberOfRvaAndSizes`, as the PE format requires.
fn data_directory(bytes: &[u8], index: usize) -> Result<(u32, u32), ()> {
    let pe_offset = read_u32(bytes, 0x3c).ok_or(())? as usize;
    let coff = pe_offset.checked_add(4).ok_or(())?;
    let optional_size = usize::from(read_u16(bytes, coff.checked_add(16).ok_or(())?).ok_or(())?);
    let optional = coff.checked_add(20).ok_or(())?;
    let optional_end = optional.checked_add(optional_size).ok_or(())?;
    if optional
        .checked_add(OPTIONAL_FIXED_LEN)
        .is_none_or(|end| end > optional_end)
    {
        return Err(());
    }
    let entries = read_u32(bytes, optional.checked_add(108).ok_or(())?).ok_or(())?;
    if index >= entries as usize {
        return Ok((0, 0));
    }
    let entry = optional
        .checked_add(OPTIONAL_FIXED_LEN)
        .and_then(|start| start.checked_add(index.checked_mul(DIRECTORY_ENTRY_LEN)?))
        .ok_or(())?;
    if entry
        .checked_add(DIRECTORY_ENTRY_LEN)
        .is_none_or(|end| end > optional_end)
    {
        return Err(());
    }
    let rva = read_u32(bytes, entry).ok_or(())?;
    let size = read_u32(bytes, entry.checked_add(4).ok_or(())?).ok_or(())?;
    Ok((rva, size))
}

/// Translate an RVA only to bytes backed by the image file. Header RVAs map to
/// the header bytes themselves. Section RVAs use the checked raw-data rule in
/// `ParsedPe`: bytes past `SizeOfRawData` are virtual zero-fill, not backed
/// file bytes, even when `VirtualSize` covers them.
fn backed_file_offset(pe: &ParsedPe, bytes: &[u8], rva: u32) -> Option<usize> {
    let at = rva as usize;
    if rva < pe.size_of_headers {
        return bytes.get(at).map(|_| at);
    }
    pe.rva_to_file_offset(rva)
        .filter(|offset| bytes.get(*offset).is_some())
}

struct ExportIndex {
    by_name: BTreeMap<String, usize>,
    by_ordinal: BTreeMap<u32, usize>,
}

fn export_index(node: &ModuleNode) -> Option<ExportIndex> {
    if node.export_error.is_some() {
        return None;
    }
    let table = node.exports.as_ref()?;
    let mut by_name = BTreeMap::new();
    let mut by_ordinal = BTreeMap::new();
    for (position, entry) in table.entries.iter().enumerate() {
        by_ordinal.insert(entry.ordinal, position);
        for name in &entry.names {
            by_name.insert(name.clone(), position);
        }
    }
    Some(ExportIndex {
        by_name,
        by_ordinal,
    })
}

fn import_export_key(name: &ImportName) -> ExportKey {
    match name {
        ImportName::Name(name) => ExportKey::Name(name.clone()),
        ImportName::Ordinal(ordinal) => ExportKey::Ordinal(u32::from(*ordinal)),
    }
}

fn resolve_import(
    modules: &[ModuleNode],
    export_indexes: &[Option<ExportIndex>],
    package_modules: &BTreeMap<String, usize>,
    consumer: usize,
    import: &ImportEntry,
) -> ImportResolution {
    let normalized = normalize_module_reference(&import.module);
    let key = import_export_key(&import.name);
    let (forwarder_chain, outcome) = if has_module_path_separator(&import.module) {
        (
            Vec::new(),
            ImportOutcome::Unresolved(UnresolvedImportReason::UnsupportedModulePath {
                module: import.module.clone(),
            }),
        )
    } else if is_api_set_reference(&import.module) {
        (
            Vec::new(),
            ImportOutcome::Unresolved(UnresolvedImportReason::ApiSetContract {
                contract: normalized.clone(),
            }),
        )
    } else if let Some(&provider) = package_modules.get(&normalized) {
        resolve_export(modules, export_indexes, package_modules, provider, key)
    } else {
        (
            Vec::new(),
            ImportOutcome::Unresolved(UnresolvedImportReason::MissingModule {
                module: normalized.clone(),
            }),
        )
    };
    ImportResolution {
        consumer: module_reference(modules, consumer),
        requested_module: import.module.clone(),
        normalized_module: normalized,
        symbol: import.name.clone(),
        iat_rva: import.iat_rva,
        forwarder_chain,
        outcome,
    }
}

fn resolve_export(
    modules: &[ModuleNode],
    export_indexes: &[Option<ExportIndex>],
    package_modules: &BTreeMap<String, usize>,
    initial_provider: usize,
    initial_key: ExportKey,
) -> (Vec<ForwarderStep>, ImportOutcome) {
    let mut provider_index = initial_provider;
    let mut key = initial_key;
    let mut chain = Vec::new();
    let mut seen = BTreeSet::new();
    seen.insert((provider_index, key.clone()));

    loop {
        let provider = module_reference(modules, provider_index);
        let node = &modules[provider_index];
        if node.export_error.is_some() {
            return (
                chain,
                ImportOutcome::Unresolved(UnresolvedImportReason::ProviderExportsMalformed {
                    provider,
                }),
            );
        }
        let entry = export_position(node, export_indexes.get(provider_index), &key)
            .and_then(|position| node.exports.as_ref()?.entries.get(position));
        let Some(entry) = entry else {
            return (
                chain,
                ImportOutcome::Unresolved(UnresolvedImportReason::MissingExport {
                    provider,
                    export: key,
                }),
            );
        };

        match &entry.target {
            ExportTarget::Empty => {
                return (
                    chain,
                    ImportOutcome::Unresolved(UnresolvedImportReason::EmptyExportSlot {
                        provider,
                        export: key,
                        ordinal: entry.ordinal,
                    }),
                );
            }
            ExportTarget::ExportRva(rva) => {
                return (
                    chain,
                    ImportOutcome::Resolved {
                        provider,
                        export: key,
                        ordinal: entry.ordinal,
                        rva: *rva,
                    },
                );
            }
            ExportTarget::Forwarder { target, .. } => {
                let parsed = parse_forwarder(target);
                let (target_module, target_export) = match parsed {
                    Ok(parsed) => parsed,
                    Err(error) => {
                        return (
                            chain,
                            ImportOutcome::Unresolved(UnresolvedImportReason::MalformedForwarder {
                                provider,
                                export: key,
                                target: target.clone(),
                                error,
                            }),
                        );
                    }
                };
                let step = ForwarderStep {
                    provider,
                    export: key.clone(),
                    target: target.clone(),
                    target_module,
                    target_export: target_export.clone(),
                };
                if is_api_set_reference(&step.target_module) {
                    return (
                        chain,
                        ImportOutcome::Unresolved(
                            UnresolvedImportReason::ForwarderApiSetContract { step },
                        ),
                    );
                }
                let Some(next_provider) = package_modules.get(&step.target_module).copied() else {
                    return (
                        chain,
                        ImportOutcome::Unresolved(UnresolvedImportReason::ForwarderMissingModule {
                            step,
                        }),
                    );
                };
                let next = (next_provider, target_export);
                if seen.contains(&next) {
                    return (
                        chain,
                        ImportOutcome::Unresolved(UnresolvedImportReason::ForwarderCycle { step }),
                    );
                }
                if chain.len() >= MAX_FORWARDER_HOPS {
                    return (
                        chain,
                        ImportOutcome::Unresolved(UnresolvedImportReason::ForwarderHopLimit {
                            step,
                        }),
                    );
                }
                seen.insert(next.clone());
                chain.push(step);
                provider_index = next_provider;
                key = next.1;
            }
        }
    }
}

fn export_position(
    node: &ModuleNode,
    index: Option<&Option<ExportIndex>>,
    key: &ExportKey,
) -> Option<usize> {
    node.exports.as_ref()?;
    let index = index?.as_ref()?;
    match key {
        ExportKey::Name(name) => index.by_name.get(name).copied(),
        ExportKey::Ordinal(ordinal) => index.by_ordinal.get(ordinal).copied(),
    }
}

fn parse_forwarder(target: &str) -> Result<(String, ExportKey), ForwarderParseError> {
    let Some((module, symbol)) = target.rsplit_once('.') else {
        return Err(ForwarderParseError::MissingSeparator);
    };
    if module.is_empty() {
        return Err(ForwarderParseError::EmptyModuleName);
    }
    if has_module_path_separator(module) {
        return Err(ForwarderParseError::ModulePathSeparator);
    }
    if symbol.is_empty() {
        return Err(ForwarderParseError::EmptySymbolName);
    }
    let export = if let Some(ordinal) = symbol.strip_prefix('#') {
        if ordinal.is_empty() {
            return Err(ForwarderParseError::EmptyOrdinal);
        }
        if !ordinal.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ForwarderParseError::InvalidOrdinalDigit);
        }
        let ordinal = ordinal
            .parse::<u32>()
            .map_err(|_| ForwarderParseError::OrdinalOverflow)?;
        ExportKey::Ordinal(ordinal)
    } else {
        ExportKey::Name(symbol.to_owned())
    };
    Ok((normalize_module_reference(module), export))
}

fn terminal_forwarder_step(reason: &UnresolvedImportReason) -> Option<&ForwarderStep> {
    match reason {
        UnresolvedImportReason::ForwarderMissingModule { step }
        | UnresolvedImportReason::ForwarderApiSetContract { step }
        | UnresolvedImportReason::ForwarderCycle { step }
        | UnresolvedImportReason::ForwarderHopLimit { step } => Some(step),
        _ => None,
    }
}

/// Add only forwarder dependencies reached while resolving an observed import.
/// A forwarder that no import uses is export inventory, not evidence needed to
/// order these statically observed dependencies.
fn add_resolution_forwarder_edges(
    resolution: &ImportResolution,
    package_modules: &BTreeMap<String, usize>,
    edges: &mut BTreeSet<(usize, usize)>,
) {
    let terminal = match &resolution.outcome {
        ImportOutcome::Unresolved(reason) => terminal_forwarder_step(reason),
        ImportOutcome::Resolved { .. } => None,
    };
    for step in resolution.forwarder_chain.iter().chain(terminal) {
        if is_api_set_reference(&step.target_module) {
            continue;
        }
        if let Some(&provider) = package_modules.get(&step.target_module) {
            edges.insert((step.provider.index, provider));
        }
    }
}

fn adjacency_lists(count: usize, edges: &BTreeSet<(usize, usize)>) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); count];
    for &(consumer, provider) in edges {
        adjacency[consumer].push(provider);
    }
    adjacency
}

fn strongly_connected_components(count: usize, adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut reverse = vec![Vec::new(); count];
    for (consumer, providers) in adjacency.iter().enumerate() {
        for &provider in providers {
            reverse[provider].push(consumer);
        }
    }

    // Iterative Kosaraju first pass: produce postorder without recursion.
    let mut visited = vec![false; count];
    let mut postorder = Vec::with_capacity(count);
    for start in 0..count {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((node, next)) = stack.last_mut() {
            if *next == adjacency[*node].len() {
                postorder.push(*node);
                stack.pop();
                continue;
            }
            let neighbor = adjacency[*node][*next];
            *next += 1;
            if !visited[neighbor] {
                visited[neighbor] = true;
                stack.push((neighbor, 0));
            }
        }
    }

    // Second pass on the reverse graph in reverse postorder.
    let mut component_of = vec![None; count];
    let mut components = Vec::new();
    for &start in postorder.iter().rev() {
        if component_of[start].is_some() {
            continue;
        }
        let id = components.len();
        let mut component = Vec::new();
        let mut stack = vec![start];
        component_of[start] = Some(id);
        while let Some(node) = stack.pop() {
            component.push(node);
            for &predecessor in &reverse[node] {
                if component_of[predecessor].is_none() {
                    component_of[predecessor] = Some(id);
                    stack.push(predecessor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

fn order_components(
    count: usize,
    components: &[Vec<usize>],
    adjacency: &[Vec<usize>],
) -> Vec<DependencyComponent> {
    let mut component_of = vec![0; count];
    for (component, members) in components.iter().enumerate() {
        for &member in members {
            component_of[member] = component;
        }
    }

    // Edges point consumer -> provider. A component can be emitted only after
    // every provider component it depends on has been emitted.
    let mut provider_dependencies = vec![BTreeSet::new(); components.len()];
    for consumer in 0..count {
        for &provider in &adjacency[consumer] {
            let consumer_component = component_of[consumer];
            let provider_component = component_of[provider];
            if consumer_component != provider_component {
                provider_dependencies[consumer_component].insert(provider_component);
            }
        }
    }

    let mut remaining: BTreeSet<_> = (0..components.len()).collect();
    let mut emitted = BTreeSet::new();
    let mut order = Vec::with_capacity(components.len());
    while !remaining.is_empty() {
        let ready = remaining
            .iter()
            .filter(|component| {
                provider_dependencies[**component]
                    .iter()
                    .all(|provider| emitted.contains(provider))
            })
            .min_by_key(|component| components[**component][0])
            .copied();
        // The condensation of strongly connected components is acyclic, so a
        // ready component must exist. Do not invent an order if that internal
        // invariant is ever broken.
        let component = ready.expect("component condensation must have a ready component");
        remaining.remove(&component);
        emitted.insert(component);

        let members = &components[component];
        let self_dependent = members.len() == 1 && adjacency[members[0]].contains(&members[0]);
        if members.len() > 1 || self_dependent {
            order.push(DependencyComponent::Circular(members.clone()));
        } else {
            order.push(DependencyComponent::Acyclic(members[0]));
        }
    }
    order
}
