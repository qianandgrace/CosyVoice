"""Compare CosyVoice's three text-normalization frontends on tricky inputs.

CosyVoiceFrontEnd degrades ttsfrd -> wetext -> built-in rules, but nothing makes
the three directly comparable, and the built-in path is easy to mistake for "no
normalization at all". This forces each one in turn by mutating
CosyVoiceFrontEnd.text_frontend on a single loaded model, then reports what each
produces for number/date/unit-heavy inputs. With --synthesize it also renders
audio per mode so the differences can actually be heard.

    python docker/vllm/scripts/compare_frontend.py
    python docker/vllm/scripts/compare_frontend.py --synthesize
    python docker/vllm/scripts/compare_frontend.py --modes ttsfrd,builtin --synthesize

Note that "wetext" and "builtin" both run the built-in cleanup (whitespace,
corner marks, trailing punctuation, paragraph splitting); wetext just runs its
own FST grammar on top first. "builtin" is therefore what you get out of the box
with no ttsfrd and no wetext FSTs.
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, 'third_party', 'Matcha-TTS'))

import torch  # noqa: E402
import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.file_utils import logging  # noqa: E402

ALL_MODES = ('ttsfrd', 'wetext', 'builtin')

# Inputs chosen to stress the parts of text normalization a plain rule set gets
# wrong: currency/decimals, dates, percentages, units, phone numbers, ordinals,
# fractions, Latin abbreviations mixed into Chinese, ranges and signs, and
# polyphonic characters that depend on disambiguation.
SCENARIOS = [
    ('money', '这件商品原价￥1,299.50，现在打8.5折，一共是1万零300元。'),
    ('date', '会议定于2024年10月14日下午3点30分召开，请提前15分钟到场。'),
    ('percent', 'GDP同比增长5.2%，总量达到126万亿元，占比约1/4。'),
    ('phone', '请拨打电话13800138000，转分机8021，或者拨打010-88886666。'),
    ('ordinal', '第3季度排名第1，出货量是去年的2.5倍。'),
    ('mixed_en', '请用USB-C接口连接iPhone 15 Pro，电池还剩20%。'),
    ('range_sign', '温度范围-10℃到35℃，湿度≤60%，误差±0.5mm。'),
    ('polyphone', '他还了钱以后还很高兴，银行行长说这样很行。'),
    ('time', '09:15出发，18:45到达，全程约2小时30分钟。'),
    ('version', '升级到v2.11.24版本后，内存占用降到512MB。'),
]


def resolve_wetext_dir():
    d = os.environ.get('COSYVOICE_WETEXT_DIR', '').strip()
    if d:
        return d
    return os.path.join(REPO_ROOT, 'pretrained_models', 'wetext')


def build_ttsfrd():
    import ttsfrd
    resource = os.path.join(REPO_ROOT, 'pretrained_models', 'CosyVoice-ttsfrd', 'resource')
    frd = ttsfrd.TtsFrontendEngine()
    assert frd.initialize(resource) is True, 'failed to initialize ttsfrd resource at {}'.format(resource)
    frd.set_lang_type('pinyinvg')
    return frd


def build_wetext():
    from wetext import Normalizer as ZhNormalizer
    from wetext import Normalizer as EnNormalizer
    d = resolve_wetext_dir()
    zh = ZhNormalizer(tagger_path='{}/zh/tn/tagger.fst'.format(d),
                      verbalizer_path='{}/zh/tn/verbalizer.fst'.format(d),
                      lang='zh')
    en = EnNormalizer(tagger_path='{}/en/tn/tagger.fst'.format(d),
                      verbalizer_path='{}/en/tn/verbalizer.fst'.format(d),
                      lang='en')
    return zh, en


class FrontendSwitcher:
    """Flips a live CosyVoiceFrontEnd between the three frontend modes.

    Building an engine is expensive (ttsfrd loads ~480 MB of resources), so each
    one is constructed at most once and cached.
    """

    def __init__(self, frontend):
        self.frontend = frontend
        self._cache = {}
        self.errors = {}

    def active_frontend(self, mode):
        """Returns the mode string text_normalize will actually branch on."""
        if mode in self.errors:
            raise RuntimeError(self.errors[mode])
        try:
            if mode == 'ttsfrd':
                if 'ttsfrd' not in self._cache:
                    self._cache['ttsfrd'] = build_ttsfrd()
                self.frontend.frd = self._cache['ttsfrd']
                self.frontend.text_frontend = 'ttsfrd'
            elif mode == 'wetext':
                if 'wetext' not in self._cache:
                    self._cache['wetext'] = build_wetext()
                self.frontend.zh_tn_model, self.frontend.en_tn_model = self._cache['wetext']
                self.frontend.text_frontend = 'wetext'
            elif mode == 'builtin':
                self.frontend.text_frontend = ''
            else:
                raise ValueError('unknown mode {}'.format(mode))
        except Exception as e:
            self.errors[mode] = '{} unavailable: {}'.format(mode, e)
            raise RuntimeError(self.errors[mode])
        return self.frontend.text_frontend


def normalize_text(switcher, text, modes):
    out = {}
    for mode in modes:
        try:
            switcher.active_frontend(mode)
            chunks = switcher.frontend.text_normalize(text, split=True)
            out[mode] = ''.join(chunks)
        except Exception as e:
            out[mode] = '<{}>'.format(e)
    return out


def print_text_report(rows, modes):
    for name, text, results in rows:
        print()
        print('=' * 100)
        print('[{}] {}'.format(name, text))
        print('-' * 100)
        for mode in modes:
            print('  {:<8}| {}'.format(mode, results[mode]))


def write_text_report(path, rows, modes):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('scenario\tinput\t{}\n'.format('\t'.join(modes)))
        for name, text, results in rows:
            cells = [results[m] for m in modes]
            f.write('{}\t{}\t{}\n'.format(name, text, '\t'.join(cells)))
    logging.info('wrote text report to %s', path)


def slugify(name):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)


def synthesize(cosyvoice, switcher, rows, modes, prompt_text, prompt_wav, out_dir):
    for mode in modes:
        try:
            switcher.active_frontend(mode)
        except Exception as e:
            logging.warning('skip synthesis for %s: %s', mode, e)
            continue
        mode_dir = os.path.join(out_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        for name, text, _ in rows:
            t0 = time.time()
            chunks = list(cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=False))
            if len(chunks) == 0:
                logging.warning('%s/%s produced no audio', mode, name)
                continue
            speech = torch.cat([c['tts_speech'] for c in chunks], dim=1)
            path = os.path.join(mode_dir, '{}_{}.wav'.format(slugify(name), len(chunks)))
            torchaudio.save(path, speech, cosyvoice.sample_rate)
            duration = speech.shape[1] / cosyvoice.sample_rate
            elapsed = time.time() - t0
            logging.info('%-8s %-10s %5.2fs audio, rtf=%.2f -> %s', mode, name, duration, elapsed / duration, path)


def main():
    parser = argparse.ArgumentParser(description='Compare CosyVoice text-normalization frontends.')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--modes', type=str, default=','.join(ALL_MODES),
                        help='comma separated subset of {}'.format(','.join(ALL_MODES)))
    parser.add_argument('--limit', type=int, default=0, help='only use the first N scenarios (0 = all)')
    parser.add_argument('--synthesize', action='store_true', help='also render audio per mode (slow)')
    parser.add_argument('--vllm', action='store_true', help='load the LLM through vLLM')
    parser.add_argument('--prompt_text', type=str,
                        default='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。')
    parser.add_argument('--prompt_wav', type=str, default='asset/zero_shot_prompt.wav')
    parser.add_argument('--out_dir', type=str, default='output/frontend_compare')
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for m in modes:
        if m not in ALL_MODES:
            raise SystemExit('unknown mode {}, pick from {}'.format(m, ALL_MODES))

    rows_input = SCENARIOS[:args.limit] if args.limit > 0 else SCENARIOS

    logging.info('loading model %s (vllm=%s)', args.model_dir, args.vllm)
    cosyvoice = AutoModel(model_dir=args.model_dir, load_vllm=args.vllm, fp16=False)
    switcher = FrontendSwitcher(cosyvoice.frontend)

    rows = []
    for name, text in rows_input:
        rows.append((name, text, normalize_text(switcher, text, modes)))
    print_text_report(rows, modes)

    os.makedirs(args.out_dir, exist_ok=True)
    write_text_report(os.path.join(args.out_dir, 'text_report.tsv'), rows, modes)

    if args.synthesize:
        if not os.path.exists(args.prompt_wav):
            raise SystemExit('prompt wav not found: {}'.format(args.prompt_wav))
        logging.info('synthesizing with prompt %s', args.prompt_wav)
        synthesize(cosyvoice, switcher, rows, modes, args.prompt_text, args.prompt_wav, args.out_dir)
        logging.info('audio written under %s/<mode>/', args.out_dir)

    logging.info('done, cuda peak memory %.2f GB', torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0)


if __name__ == '__main__':
    main()
