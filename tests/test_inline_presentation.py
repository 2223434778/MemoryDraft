from types import SimpleNamespace

import pytest

from memorydraft.serving.inline import InlinePresentation, utf16_length


class Tokenizer:
    def decode(self, ids, **kwargs):
        return bytes(i for i in ids if i != 256).decode('utf-8', errors='replace')


def observer(tokenizer=None):
    return InlinePresentation(SimpleNamespace(tokenizer=tokenizer or Tokenizer()))


def draft(ids, tree=None):
    return {'phase':'draft', 'proposal_ids':ids, 'tree':tree}


def verified(ids, accepted):
    return {'phase':'verified', 'committed_ids':ids, 'accepted':accepted}


def test_contextual_preview_and_acceptance_recovery_ranges():
    view = observer(); prefix = list(b'Hi ')
    assert view.present(draft(list(b'goodbye')), prefix)['candidate'] == 'goodbye'
    result = view.present(verified(list(b'good!'), 4), prefix)
    assert result == {'start':3, 'accepted_end':7, 'end':8, 'accepted':'good', 'target':'!', 'rejected':'bye'}


def test_target_byte_completing_chinese_is_not_falsely_attributed_to_draft():
    view = observer(); prefix = [0xe4]
    assert view.present(draft([0xbd, 0xa0]), prefix)['candidate'] == '你'
    result = view.present(verified([0xbd, 0xa0], 1), prefix)
    assert result['start'] == result['accepted_end'] == 0
    assert result['end'] == 1 and result['accepted'] == '' and result['target'] == '你'


def test_utf16_ranges_match_browser_offsets_after_emoji():
    view = observer(); prefix = list('A😀'.encode())
    view.present(draft(list(b'BC')), prefix)
    result = view.present(verified(list(b'BC'), 1), prefix)
    assert (result['start'], result['accepted_end'], result['end']) == (3, 4, 5)
    assert utf16_length('你好😀') == 4


def test_context_cleanup_can_replace_a_tail_without_inventing_acceptance():
    class Cleanup(Tokenizer):
        def decode(self, ids, **kwargs):
            return super().decode(ids, **kwargs).replace(' .', '.')
    view = observer(Cleanup()); prefix = list(b'Hi ')
    view.present(draft(list(b'word')), prefix)
    result = view.present(verified([46], 0), prefix)
    assert result['start'] == result['accepted_end'] == 2 and result['end'] == 3
    assert result['target'] == '.'


def test_tree_preview_uses_a_real_path_and_selected_branch_can_differ():
    view = observer(); prefix = list(b'A')
    tree = {'token_ids':list(b'ABCD'), 'parents':[-1,0,1,0], 'depths':[0,1,2,1]}
    assert view.present(draft([], tree), prefix)['candidate'] == 'BC'
    result = view.present(verified(list(b'D'), 1), prefix)
    assert result['accepted'] == 'D' and result['rejected'] == 'BC'


def test_special_tokens_do_not_appear_in_answer_ranges():
    view = observer(); view.present(draft([111,107,256]), [])
    result = view.present(verified([111,107,256], 3), [])
    assert result['accepted'] == 'ok' and result['end'] == 2 and not result['rejected']


def test_previews_are_bounded_and_never_include_the_confirmed_prefix():
    view = observer(); prefix = list(b'PRIVATE EARLIER CONTENT ')
    result = view.present(draft([120]*100), prefix)
    assert len(result['candidate']) == 80
    assert 'PRIVATE' not in repr(result)
