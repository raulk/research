"""
Transaction and blob data structures.

Models type 3 (blob-carrying) transactions with cell-level granularity.
"""

from dataclasses import dataclass, field
import hashlib


# Protocol constants (from EIP-7594)
CELLS_PER_EXT_BLOB = 128
CELL_SIZE = 2048  # bytes
RECONSTRUCTION_THRESHOLD = 64  # cells needed for Reed-Solomon decoding
MAX_BLOBS_PER_TX = 6


@dataclass
class BlobCell:
    """
    A single cell from a blob (2048 bytes + proof).

    Attributes:
        blob_index: Which blob this cell belongs to
        column_index: Column index (0-127)
        data: Cell data (simulated as size only)
        proof: KZG proof (simulated)
    """
    blob_index: int
    column_index: int
    data_size: int = CELL_SIZE
    proof_size: int = 48  # KZG proof size

    @property
    def total_size(self) -> int:
        """Total size including proof."""
        return self.data_size + self.proof_size

    def __hash__(self):
        return hash((self.blob_index, self.column_index))


@dataclass
class Blob:
    """
    A blob with 128 columns, each containing a cell.

    Attributes:
        index: Index within the transaction (0-5)
        hash: Blob versioned hash
        available_columns: Set of column indices available locally
    """
    index: int
    hash: str = field(default_factory=lambda: hashlib.sha256().hexdigest())
    available_columns: set[int] = field(default_factory=set)

    def add_columns(self, columns: set[int]):
        """Add columns to available set."""
        self.available_columns.update(columns)

    def has_column(self, column_index: int) -> bool:
        """Check if a specific column is available."""
        return column_index in self.available_columns

    def has_all_columns(self) -> bool:
        """Check if all 128 columns are available."""
        return len(self.available_columns) == CELLS_PER_EXT_BLOB

    def can_reconstruct(self) -> bool:
        """Check if we have enough columns for Reed-Solomon reconstruction."""
        return len(self.available_columns) >= RECONSTRUCTION_THRESHOLD

    def get_cell(self, column_index: int) -> BlobCell | None:
        """Get a cell if available."""
        if not self.has_column(column_index):
            return None
        return BlobCell(blob_index=self.index, column_index=column_index)

    def get_cells(self, column_indices: set[int]) -> list[BlobCell]:
        """Get multiple cells."""
        return [
            BlobCell(blob_index=self.index, column_index=idx)
            for idx in column_indices
            if self.has_column(idx)
        ]

    @property
    def cell_mask(self) -> int:
        """
        Get cell_mask as uint128 bitarray.

        Returns a bitmask where bit i is set if column i is available.
        """
        mask = 0
        for col in self.available_columns:
            mask |= (1 << col)
        return mask

    @staticmethod
    def from_cell_mask(index: int, mask: int, hash: str | None = None) -> "Blob":
        """
        Create a Blob from a cell_mask bitarray.

        Args:
            index: Blob index within transaction
            mask: uint128 bitmask of available columns
            hash: Optional blob hash

        Returns:
            Blob instance with columns set from mask
        """
        columns = set()
        for i in range(CELLS_PER_EXT_BLOB):
            if mask & (1 << i):
                columns.add(i)

        return Blob(
            index=index,
            hash=hash or hashlib.sha256().hexdigest(),
            available_columns=columns
        )

    @property
    def size(self) -> int:
        """Size in bytes of available data."""
        return len(self.available_columns) * CELL_SIZE


@dataclass
class Transaction:
    """
    A type 3 (blob-carrying) transaction.

    Attributes:
        hash: Transaction hash
        sender: Sender address
        nonce: Transaction nonce
        blobs: List of blobs attached to this transaction
        timestamp: When transaction was created (milliseconds)
        gas_fee: Priority fee for ordering
    """
    hash: str = field(default_factory=lambda: hashlib.sha256().hexdigest())
    sender: str = "0x0000000000000000000000000000000000000000"
    nonce: int = 0
    blobs: list[Blob] = field(default_factory=list)
    timestamp: float = 0.0
    gas_fee: float = 0.0

    def __post_init__(self):
        """Validate transaction."""
        if len(self.blobs) > MAX_BLOBS_PER_TX:
            raise ValueError(f"Transaction can have at most {MAX_BLOBS_PER_TX} blobs")

    def add_blob(self, blob: Blob):
        """Add a blob to the transaction."""
        if len(self.blobs) >= MAX_BLOBS_PER_TX:
            raise ValueError(f"Transaction already has {MAX_BLOBS_PER_TX} blobs")
        self.blobs.append(blob)

    def has_full_availability(self) -> bool:
        """Check if all blobs have full availability (all 128 columns)."""
        return all(blob.has_all_columns() for blob in self.blobs)

    def can_reconstruct_all(self) -> bool:
        """Check if all blobs can be reconstructed."""
        return all(blob.can_reconstruct() for blob in self.blobs)

    def get_common_cell_mask(self) -> int:
        """
        Get cell_mask representing columns available for ALL blobs.

        This is used in NewPooledTransactionHashes announcements.
        """
        if not self.blobs:
            return 0

        # Intersection of all blob masks
        mask = self.blobs[0].cell_mask
        for blob in self.blobs[1:]:
            mask &= blob.cell_mask

        return mask

    @property
    def total_size(self) -> int:
        """Total size of all blob data in bytes."""
        return sum(blob.size for blob in self.blobs)

    @property
    def blob_count(self) -> int:
        """Number of blobs."""
        return len(self.blobs)

    def __hash__(self):
        return hash(self.hash)

    def __eq__(self, other):
        if not isinstance(other, Transaction):
            return False
        return self.hash == other.hash


def create_full_transaction(
    num_blobs: int = 1,
    sender: str = "0x0000000000000000000000000000000000000000",
    nonce: int = 0,
    timestamp: float = 0.0
) -> Transaction:
    """
    Create a transaction with full blob availability.

    Args:
        num_blobs: Number of blobs (1-6)
        sender: Sender address
        nonce: Transaction nonce
        timestamp: Creation timestamp

    Returns:
        Transaction with all columns available for all blobs
    """
    if num_blobs < 1 or num_blobs > MAX_BLOBS_PER_TX:
        raise ValueError(f"num_blobs must be between 1 and {MAX_BLOBS_PER_TX}")

    tx = Transaction(sender=sender, nonce=nonce, timestamp=timestamp)

    for i in range(num_blobs):
        blob = Blob(index=i)
        # Add all 128 columns
        blob.add_columns(set(range(CELLS_PER_EXT_BLOB)))
        tx.add_blob(blob)

    return tx


def create_sampled_transaction(
    num_blobs: int,
    custody_columns: set[int],
    sender: str = "0x0000000000000000000000000000000000000000",
    nonce: int = 0,
    timestamp: float = 0.0
) -> Transaction:
    """
    Create a transaction with only custody columns available.

    Args:
        num_blobs: Number of blobs (1-6)
        custody_columns: Set of column indices in custody
        sender: Sender address
        nonce: Transaction nonce
        timestamp: Creation timestamp

    Returns:
        Transaction with only custody columns available
    """
    if num_blobs < 1 or num_blobs > MAX_BLOBS_PER_TX:
        raise ValueError(f"num_blobs must be between 1 and {MAX_BLOBS_PER_TX}")

    tx = Transaction(sender=sender, nonce=nonce, timestamp=timestamp)

    for i in range(num_blobs):
        blob = Blob(index=i)
        blob.add_columns(custody_columns)
        tx.add_blob(blob)

    return tx
