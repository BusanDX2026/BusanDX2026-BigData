"""Patch only the verified background paragraph; retain HWPX styles and references."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree

SOURCE = Path('C:/Users/wjdck/Downloads/분야1. 빅데이터 분석 및 시각화_수문장.hwpx')
DEST = Path('C:/project_git/BusanDX2026-BigData/output/hwpx/분야1. 빅데이터 분석 및 시각화_수문장_배경근거수정.hwpx')
OLD = ' - 최근 남부지역에서는 집중호우로 인한 침수와 시설 피해가 반복되고 있습니다. 2024년 9월 전남·경남을 중심으로 발생한 호우의 전체 재산 피해는 711억 원, 복구비는 1,137억 원으로 확정되었습니다. 2025년 7월에는 산청·합천 등 경남지역의 호우 피해액이 5,177억 원에 달했으며, 복구에 1조 1,947억 원이 투입되었습니다.'
FIRST = ' - 최근 남부지역에서는 집중호우로 인한 침수와 시설 피해가 반복되고 있습니다. 2024년 9월 19~21일 호우로 전남·경남권 등을 중심으로 총 711억 원의 재산 피해가 발생했으며, 복구비는 1,137억 원으로 확정되었습니다.'
SECOND = ' 2025년 7월에는 산청·합천 등 경남지역에서 5,177억 원의 호우 피해가 발생했고, 복구비는 1조 1,947억 원으로 확정되었습니다.'
old_fragment = f'<hp:run charPrIDRef="44"><hp:t>{OLD}</hp:t></hp:run><hp:run charPrIDRef="39"><hp:t>[7, 8] </hp:t></hp:run>'
new_fragment = f'<hp:run charPrIDRef="44"><hp:t>{FIRST}</hp:t></hp:run><hp:run charPrIDRef="39"><hp:t>[7]</hp:t></hp:run><hp:run charPrIDRef="44"><hp:t>{SECOND}</hp:t></hp:run><hp:run charPrIDRef="39"><hp:t>[8] </hp:t></hp:run>'

with ZipFile(SOURCE) as original:
    data = {i.filename: original.read(i.filename) for i in original.infolist()}
    section = data['Contents/section0.xml'].decode('utf-8')
    assert section.count(old_fragment) == 1, 'Expected exactly one target paragraph'
    changed = section.replace(old_fragment, new_fragment)
    etree.fromstring(changed.encode('utf-8'))
    assert changed.replace(new_fragment, old_fragment) == section
    data['Contents/section0.xml'] = changed.encode('utf-8')
    preview = data['Preview/PrvText.txt'].decode('utf-8')
    if OLD in preview:
        assert OLD + '[7, 8]' in preview
        data['Preview/PrvText.txt'] = preview.replace(OLD + '[7, 8]', FIRST + '[7]' + SECOND + '[8]').encode('utf-8')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    assert not DEST.exists(), 'Do not overwrite an existing output'
    with ZipFile(DEST, 'w') as output:
        for info in original.infolist():
            output.writestr(info, data[info.filename])
        output.comment = original.comment

with ZipFile(SOURCE) as original, ZipFile(DEST) as output:
    assert output.testzip() is None
    assert original.namelist() == output.namelist()
    differences = [name for name in output.namelist() if original.read(name) != output.read(name)]
    assert set(differences) <= {'Contents/section0.xml', 'Preview/PrvText.txt'}
    for name in output.namelist():
        if name.endswith(('.xml', '.hpf', '.rdf')):
            etree.fromstring(output.read(name))
    print('Changed entries:', differences)
    print('Verified: other content, references, style definitions and images unchanged.')
    print('New paragraph:', FIRST + '[7]' + SECOND + '[8]')
    print('Output:', DEST)
