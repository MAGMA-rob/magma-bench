"""Check that bench's serialized instruction boundary matches core."""
import pytest

from magma_core.simulation.data_structures import Instruction, UserInstruction, StatusReturn
from magma_bench.artifacts.models import InstructionSpec


@pytest.mark.parametrize('instruction', [UserInstruction('Do it'), StatusReturn({'infos': 'Done'})])
def test_instruction_round_trip_through_benchmark_schema(instruction):
    spec = InstructionSpec.model_validate(instruction.to_spec())
    loaded = InstructionSpec.model_validate_json(spec.model_dump_json())
    restored = Instruction.from_spec(loaded.model_dump())
    assert type(restored) is type(instruction)
    assert restored.get_content() == instruction.get_content()
