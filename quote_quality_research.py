# SETUP LOGIC: 模块定义与已有依赖；导入本身不表示计算或训练已完成。
"""Small add-on controls for completed Step5 work; importing never loads or fits.

Call show_research(globals()) in the existing idle Step5 kernel. Saved-only
sessions can review predictions without loading raw quotes or event history.
"""
# SETUP LOGIC: 有限扩展的显示/保存依赖；导入不加载quote或拟合模型。
from html import escape
from io import BytesIO
import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# VALIDATION LOGIC: 拒绝busy kernel；优先已完成live结果，否则读已保存validation。
def research_context(namespace=None, folder='outputs/quote_quality_step5'):
    """Resolve completed results without executing the Step5 script or loader."""
    from quote_quality_saved import load_saved_validation, validation_rows
    ns = namespace if namespace is not None else {}
    if ns.get('step5_busy') or ns.get('step5_research_busy'):
        raise ValueError('Step5 is busy. Keep its kernel and wait for the current action.')
    frame, predictions = ns.get('step5_frame'), ns.get('step5_predictions')
    live = frame is not None and predictions is not None
    if live:
        metadata = dict(ns.get('step5_result_metadata') or {})
        source = 'Completed Step5 objects in this kernel'
    else:
        frame, predictions, metadata = load_saved_validation(folder)
        source = 'Saved completed validation; raw quotes were not loaded'
    # VALIDATION LOGIC: 合并既有test/locked状态，不能通过打开新控件解除锁。
    # CORE LOGIC: STEP 1 — 合并现有实验锁且不清除已有暴露记录
    # Input: metadata={test_available:True,locked_choice:None}; live=True; ns step5_locked=Age decay,step5_test_predictions为空。
    # Output: metadata保留test_available=True，locked_choice=Age decay，saved_lock_guard=False（两来源均无guard）。
    # Trick: setdefault不以当前空表覆盖已保存的True；显式解锁不是打开新控件的副作用。
    metadata = dict(metadata)
    if ns.get('step5_locked') is not None:
        metadata['locked_choice'] = ns['step5_locked']
    if ns.get('step5_test_predictions') is not None and not ns['step5_test_predictions'].empty:
        metadata['test_available'] = True
    if live and 'step5_locked' in ns:
        metadata.setdefault('locked_choice', ns['step5_locked'])
    if live and 'step5_test_predictions' in ns:
        metadata.setdefault('test_available', ns['step5_test_predictions'] is not None and not ns['step5_test_predictions'].empty)
    metadata['saved_lock_guard'] = bool(metadata.get('saved_lock_guard') or ns.get('step5_saved_lock_guard'))
    # CORE LOGIC: STEP 2 — 校验同Validation队列后返回原对象引用
    # Input: 完整frame含Validation row_ids=[7,8]，predictions每模型均含[7,8]；live=True，ns.step5_event_cache=E。
    # Output: 返回frame/predictions原引用、已合并metadata及event_cache=E；live=False则event_cache=None。
    # Trick: validation_rows先核对队列；此处不复制大frame或重新加载raw。
    validation_rows(frame, predictions)
    return dict(frame=frame, predictions=predictions, metadata=metadata,
                event_cache=ns.get('step5_event_cache') if live else None, source=source)


# SETUP LOGIC: movement_coverage_figure：函数接口；计算阶段见内部CORE标记。
def movement_coverage_figure(frame, sidecar, quote_dates=None):
    """Describe finite direction signals, not the availability of zero counts."""
    # CORE LOGIC: STEP 1 — 同row_id对齐Train/Validation信号
    # Input: frame rows=(1,Train,issuer_count0),(2,Test,0),(3,Validation,2)；sidecar row1/3分别有direction1/NaN.
    # Output: rows只含1和3；direction=[1,NaN]；Test row2不进诊断。
    # Trick: merge(validate=one_to_one)拒绝重复ID造成的样本膨胀。
    from quote_quality_movement import DIRECTION_FEATURES, ISSUER_FEATURES
    keys = ['row_id', 'split', 'NUM_OF_ISSUER_TRADES_SINCE_PREV']
    rows = frame.loc[frame['split'].isin(['Train', 'Validation']), keys].merge(
        sidecar, on='row_id', how='left', validate='one_to_one')
    # CORE LOGIC: STEP 2 — 限定真实quote-file日期的覆盖分母
    # Input: quote_dates=[2026-03-02T00:00-05:00,2026-03-03T00:00-05:00]；rows time为03-01/02/03各10:00 ET。
    # Output: inside=[False,True,True]；没有日期参数时三行均True。
    # Trick: 先换纽约日期再比较；不是用UTC日期错移前夜的query。
    if quote_dates:
        dates = pd.to_datetime(pd.Series(quote_dates), utc=True).dt.tz_convert('America/New_York').dt.normalize()
        day = pd.to_datetime(rows.time, utc=True).dt.tz_convert('America/New_York').dt.normalize()
        inside = day.between(dates.iloc[0], dates.iloc[1])
    else:
        inside = pd.Series(True, index=rows.index)
    # Signal columns are specified by the movement module, not inferred from counts.
    # CORE LOGIC: STEP 3 — 只把数值信号当有支持
    # Input: bid/ask direction guarded_mean=[(1,NaN),(NaN,NaN)]；issuer mean=[(NaN,NaN),(2,NaN)]，计数均非缺失。
    # Output: flags Direction=[True,False],Issuer=[False,True]；计数0不会制造numeric支持。
    # Trick: any(axis=1)表示至少一侧存在信号，不要求双边齐全。
    direction = [c for c in DIRECTION_FEATURES if c.endswith('_move_guarded_mean_bps')]
    issuer = [c for c in ISSUER_FEATURES if c.endswith('_issuer_move_mean_bps')]
    if not direction or not issuer:
        raise ValueError('Movement module is missing its documented primary signal columns')
    flags = pd.DataFrame({
        'Direction': rows[direction].notna().any(axis=1),
        'Issuer': rows[issuer].notna().any(axis=1),
    }, index=rows.index)
    # PLOTTING LOGIC: 2×2全尺寸覆盖图与标题。
    fig = Figure(figsize=(16, 10), facecolor='white')
    a, b, c, d = fig.subplots(2, 2).flat
    fig.subplots_adjust(left=.09, right=.96, top=.86, bottom=.14, hspace=.48, wspace=.3)
    fig.suptitle('30-minute quote movement | support before fitting', fontsize=16)
    fig.text(.5, .91, f'{len(rows):,} Train/Validation targets retained; {int(inside.sum()):,} within quote-file ET dates', ha='center')
    # CORE LOGIC: STEP 4 — 按split算信号覆盖率
    # Input: inside两条Validation行，flags Direction=[True,False],Issuer=[False,False]，无Train行。
    # Output: shown Validation=(0.5,0.0),Train=(NaN,NaN).
    # Trick: 布尔mean是样本比例；不存在Train不能补0声称测得零覆盖。
    shown = flags.loc[inside].groupby(rows.loc[inside, 'split'], observed=True).mean().reindex(['Train', 'Validation'])
    # PLOTTING LOGIC: 绘Train/Validation两组覆盖条，不更改分母。
    x = np.arange(len(shown))
    for offset, name, color in [(-.18, 'Direction', '#287f8e'), (.18, 'Issuer', '#cb694d')]:
        bars = a.bar(x + offset, shown[name], .35, label=name, color=color)
        a.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    a.set_xticks(x, shown.index); a.set_ylim(0, 1.1); a.legend(fontsize=9)
    a.set_title('At least one side has a supported numeric signal'); a.set_ylabel('Fraction within quote-file dates')
    # CORE LOGIC: STEP 5 — 区分原issuer成交信息是否存在
    # Input: counts=[0,2,NaN]；Direction flags=[True,False,True]；inside均True.
    # Output: strata=[No intervening trades or missing anchor,Intervening issuer trades,Unknown]；各组Direction coverage=[1,0,1].
    # Trick: count=0也可能缺anchor；不能直接把整组解释成没有issuer交易。
    count = pd.to_numeric(rows.NUM_OF_ISSUER_TRADES_SINCE_PREV, errors='coerce')
    strata = pd.Series(np.select([count.eq(0), count.gt(0)], ['No intervening trades\nor missing anchor', 'Intervening issuer trades'], default='Unknown'), index=rows.index)
    support = flags.loc[inside].groupby(strata.loc[inside], observed=True).mean()
    # PLOTTING LOGIC: 显示分层覆盖和signed bps分布；图不声明预测增益。
    y = np.arange(len(support))
    for offset, name, color in [(-.18, 'Direction', '#287f8e'), (.18, 'Issuer', '#cb694d')]:
        b.barh(y + offset, support[name], .35, label=name, color=color)
    b.set_yticks(y, support.index, fontsize=8); b.set_xlim(0, 1.1)
    b.set_title('Coverage against existing issuer-trade information'); b.set_xlabel('Fraction; zero count also includes missing anchor')
    for ax, names, title in [(c, direction, 'Target-bond common-dealer direction'), (d, issuer, 'Other-bond equal-weight issuer direction')]:
        for col in names:
            values = pd.to_numeric(rows.loc[inside, col], errors='coerce').dropna()
            if len(values):
                ax.hist(values, bins=40, histtype='step', label=f'{col}\nn={len(values):,}')
        ax.set_title(title); ax.set_xlabel('Signed change (bps); positive = wider')
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    for ax in [a, b, c, d]: ax.grid(alpha=.15)
    fig.text(.06, .06, 'No quote / insufficient support stays unknown, not zero. Current BASE14 and target/anchor are unchanged.', fontsize=10)
    fig.text(.06, .035, 'Feature coverage does not establish predictive benefit. Each candidate needs its own comparison with saved Age decay.', fontsize=10)
    return fig


# UI LOGIC: 控件会话仅持有扩展状态，保留原Step5对象和锁。
class ResearchSession:
    """Own only add-on state. Never resets the original kernel or test lock."""
    # UI LOGIC: 初始化显示和回调；默认不触发feature build或fit。
    def __init__(self, namespace=None, folder='outputs/quote_quality_step5', output_folder='outputs/quote_quality_incremental'):
        import ipywidgets as widgets
        self.namespace = namespace if namespace is not None else {}
        context = research_context(self.namespace, folder)
        self.frame = context['frame']; self.predictions = context['predictions']
        self.original_predictions = context['predictions']; self.metadata = context['metadata']
        self.event_cache = context['event_cache']; self.source = context['source']
        self.folder = Path(folder); self.output = Path(output_folder)
        self.sidecar = None; self.sidecar_saved = False; self.rule_review = None; self.case_manifest = None
        self.busy = False; self.last_figure = None; self.last_path = None
        self.controls = []
        self._last_progress = 0.; self._stop = Event()
        self.status = widgets.HTML(value=escape(self.source))
        self.detail = widgets.HTML()
        self.elapsed = widgets.HTML()
        self.image = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1600px'))
        self.progress = widgets.IntProgress(min=0, max=1, description='Work:', layout=widgets.Layout(width='650px'))
        choices = self.predictions.model.drop_duplicates().tolist()
        self.selected = widgets.Dropdown(options=choices, value='Age decay' if 'Age decay' in choices else choices[-1], description='Compare:')
        self.buttons = {}
        actions = [('losses', 'Review losses'), ('tails', 'Review sector/tails'), ('rules', 'Review rule effects'), ('cases', 'Freeze cases'),
                   ('features', 'Build movement features'), ('Direction', 'Fit Direction (1)'), ('Issuer', 'Fit Issuer (1)')]
        for action, label in actions:
            button = widgets.Button(description=label, layout=widgets.Layout(width='185px'))
            button.on_click(lambda _, action=action: self.run(action))
            self.buttons[action] = button
        self.controls = [self.selected, *self.buttons.values()]
        self.dashboard = widgets.VBox([
            widgets.HTML('<b>Step5 finite extension</b> · Saved results first; 30-minute window fixed. Each Fit trains one new candidate.'),
            widgets.HBox([self.selected, self.buttons['losses'], self.buttons['tails']]),
            widgets.HBox([self.buttons['rules'], self.buttons['cases'], self.buttons['features']]),
            widgets.HBox([self.buttons['Direction'], self.buttons['Issuer']]),
            self.status, self.progress, self.detail, self.elapsed, self.image,
        ])
        self._restore_sidecar()
        self._enable()

    # VALIDATION LOGIC: 每次操作前读取原kernel最新锁；已有元数据的test暴露标记不被False覆盖。
    def _metadata_now(self):
        # CORE LOGIC: STEP 1 — 每次操作读取新增锁状态
        # Input: self.metadata={test_available:True,locked_choice:None}; ns.step5_locked=Age decay,step5_test_predictions为空，guard都未设置。
        # Output: result={test_available:True,locked_choice:Age decay,saved_lock_guard:False}。
        # Trick: 已有True不能由空table清零。
        result = dict(self.metadata)
        ns = self.namespace
        if ns.get('step5_locked') is not None: result['locked_choice'] = ns['step5_locked']
        if ns.get('step5_test_predictions') is not None and not ns['step5_test_predictions'].empty:
            result['test_available'] = True
        result['saved_lock_guard'] = bool(result.get('saved_lock_guard') or ns.get('step5_saved_lock_guard'))
        return result

    # VALIDATION LOGIC: 只要选定锁、看过test或saved guard任一为真即禁新fit。
    def _locked(self):
        # CORE LOGIC: STEP 1 — 任一锁或test暴露即关闭新增fit
        # Input: meta={locked_choice:None,test_available:True,saved_lock_guard:False}。
        # Output: True；若三项都无锁/False，返回False。
        meta = self._metadata_now()
        return meta.get('locked_choice') is not None or bool(meta.get('test_available') or meta.get('saved_lock_guard'))

    # CACHEING LOGIC: 仅恢复来源和校验和匹配的movement sidecar；不因缺失而自动重建。
    def _restore_sidecar(self):
        from quote_quality_incremental import load_movement_sidecar
        try:
            self.sidecar, meta = load_movement_sidecar(self.frame, self.metadata, folder=self.output)
            self.sidecar_saved = True
            self.status.value += '<br>Restored matching movement features: ' + escape(str(meta['snapshot_path']))
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as error:
            self.detail.value = escape(f'Saved movement features were not reused: {error}')

    # UI LOGIC: busy/锁/sidecar状态控制按钮；无event cache提示原kernel。
    def _enable(self):
        for control in self.controls: control.disabled = self.busy
        if self.busy: return
        self.buttons['features'].disabled = self.sidecar_saved or (self.sidecar is None and self.event_cache is None)
        for block in ['Direction', 'Issuer']:
            self.buttons[block].disabled = not self.sidecar_saved or self._locked() or f'Age decay + {block}' in set(self.predictions.model)
        if self.event_cache is None and self.sidecar is None:
            self.detail.value = 'Movement needs step5_event_cache in the original idle kernel. This panel will not rebuild event history.'
        if self._locked():
            self.detail.value = 'The existing experiment is locked or test was exposed. Reviews remain available; new fits are disabled.'

    # PROGRESS LOGIC: 节流UI刷新；完成通知不被0.2秒节流吞掉。
    def _report(self, stage, completed=None, total=None, detail=''):
        now = perf_counter()
        if completed != total and now - self._last_progress < .2: return
        self._last_progress = now
        self.progress.description = str(stage).replace('_', ' ')[:16] + ':'
        self.progress.max = max(1, int(total or 1))
        self.progress.value = min(self.progress.max, int(completed or 0))
        count = f' {completed:,}/{total:,}' if completed is not None and total is not None else ''
        self.detail.value = escape(str(detail) + count)

    # PROGRESS LOGIC: 后台计时器仅更新显示，停止Event结束线程。
    def _timer(self, started):
        while not self._stop.wait(1): self._set_elapsed(started)

    # PROGRESS LOGIC: 秒转时分秒；不估算完成比例或训练增益。
    def _set_elapsed(self, started):
        seconds = int(perf_counter() - started)
        self.elapsed.value = f'Elapsed: {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'

    # PLOTTING LOGIC: 保存完整高分辨率PNG，并以较小PNG显示同屏图。
    def _render(self, figure, slug):
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f'{slug}_{uuid4().hex[:10]}.png'
        figure.savefig(path, dpi=160)
        with BytesIO() as buffer:
            figure.savefig(buffer, format='png', dpi=105)
            self.image.value = buffer.getvalue()
        self.last_figure, self.last_path = figure, path
        return path

    # VALIDATION LOGIC: 日期必须来自原实验元数据；不从交易日期猜测quote覆盖。
    def _quote_dates(self):
        # CORE LOGIC: STEP 1 — quote覆盖日期须来自原实验
        # Input: metadata.quote_file_dates=["2026-03-01T00:00-05:00","2026-04-01T00:00-04:00"]。
        # Output: 返回原两元素列表；若改成None或仅一个日期则ValueError，不从trades猜。
        dates = self.metadata.get('quote_file_dates')
        if not isinstance(dates, (list, tuple)) or len(dates) != 2:
            raise ValueError('The saved experiment needs its actual quote_file_dates for this review; do not infer them from trades.')
        return dates

    # CACHEING LOGIC: 同会话规则诊断只算一次。
    def _rules(self):
        from quote_quality_saved import trade_rule_diagnostics
        if self.rule_review is None:
            # CORE LOGIC: STEP 1 — 只对实际Train/Validation query做规则诊断
            # Input: frame split=[Train,Validation,Test]，row_id=[1,2,3]；quote日期覆盖1和2。
            # Output: eligible=[1,2]；rule_review只用1/2，impacts附actual saved queries来源。
            # Trick: 不触碰Test；feature数值变化不等于预测误差改善。
            eligible = self.frame.loc[self.frame['split'].isin(['Train', 'Validation'])]
            self.rule_review = trade_rule_diagnostics(eligible, *self._quote_dates())
            self.rule_review['impacts'].attrs.update(
                query_source='actual saved Train/Validation trade queries within quote-file ET dates',
                start=self.rule_review['start'], end=self.rule_review['end'])
        # CACHEING LOGIC: 返回已保留的诊断对象。
        return self.rule_review

    @staticmethod
    # FILE IO LOGIC: 保存全部分组CSV及定义/限制JSON；不向notebook打印大表。
    def _export_diagnostics(result, path):
        for name in ['groups', 'daily', 'overall_daily', 'definitions']:
            result[name].to_csv(path.with_name(path.stem + '_' + name + '.csv'), index=False)
        result['overall'].to_csv(path.with_name(path.stem + '_overall.csv'), header=['value'])
        info = {name: result[name] for name in ['selected', 'reference', 'n', 'source', 'limitations']}
        path.with_suffix('.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

    # UI LOGIC: 按钮分发、防重入、计时、错误提示和finally恢复。
    def run(self, action):
        if self.busy: return
        if self.namespace.get('step5_busy') or self.namespace.get('step5_research_busy'):
            self.status.value = 'Step5 is busy; wait for its current action.'
            return
        self.busy = True; self.namespace['step5_research_busy'] = True
        self._enable(); self._stop = Event(); started = perf_counter()
        self.progress.bar_style = 'info'; self.image.value = b''
        timer = Thread(target=self._timer, args=(started,), daemon=True); timer.start()
        try:
            if action in ['losses', 'tails']:
                from quote_quality_diagnostics import diagnostics, figure, sector_tail_figure
                # CORE LOGIC: STEP 1 — 读取相同validation样本的误差差异
                # Input: 两个同row_id验证样本；selected=Age decay绝对误差bps=[1,3]，直接参考Reliability误差=[2,2]，其余原实验字段完整。
                # Output: diagnostics.n=2，平均改变量0bps，各样本改变量[-1,+1]；切片继承同一母集，不重训。
                # Trick: reference由既有模型链指定；负改变量表示误差降低。
                result = diagnostics(self.frame, self.predictions, self.selected.value)
                # PLOTTING LOGIC: 绘并保存已算losses或SECTOR/tails诊断。
                plot = figure if action == 'losses' else sector_tail_figure
                path = self._render(plot(result), 'validation_' + action)
                self._export_diagnostics(result, path)
                self.status.value = escape(f'Saved same-row validation review: {path}')
            elif action == 'rules':
                from quote_quality_saved import trade_rule_figure
                result = self._rules()
                path = self._render(trade_rule_figure(result), 'train_validation_rule_effects')
                self.status.value = escape(f'Train/Validation feature effects only. Saved {path}')
            elif action == 'cases':
                from quote_quality_diagnostics import fixed_trade_case_manifest
                # CORE LOGIC: STEP 2 — 用真实query日期冻结可复查案例
                # Input: frame rows=Train1@03-01,Validation2@03-02,Test3@03-02；rule日期03-02至03-31。
                # Output: 传给fixed_trade_case_manifest的仅row2；已有manifest传入保留，文件保存到fixed_trade_cases.csv。
                # Trick: 冻结random/typical/impact用于导航，不将病例样本当全群体发生率。
                rule = self._rules()
                frame = self.frame.loc[self.frame['split'].isin(['Train', 'Validation'])]
                day = pd.to_datetime(frame.time, utc=True).dt.tz_convert('America/New_York').dt.normalize()
                frame = frame.loc[day.between(rule['start'], rule['end'])]
                self.case_manifest = fixed_trade_case_manifest(frame, rule['impacts'], existing=self.case_manifest,
                    manifest_path=self.output / 'fixed_trade_cases.csv', quote_start=rule['start'], quote_end=rule['end'])
                # UI LOGIC: 显示冻结结果并分发其余操作；错误不会重启kernel。
                self.status.value = escape(f'Frozen {len(self.case_manifest)} cases from actual Train/Validation queries. No event aggregation was run.')
                self._show_cases()
            elif action == 'features':
                self._build()
            elif action in ['Direction', 'Issuer']:
                self._fit(action)
            else:
                raise ValueError('Unknown research action')
            self.progress.bar_style = 'success'
        except KeyboardInterrupt:
            self.status.value = 'Interrupted. Original Step5 results and fully saved add-on results are retained.'
            self.progress.bar_style = 'warning'
        except Exception as error:
            self.status.value = escape(f'{type(error).__name__}: {error}')
            self.progress.bar_style = 'danger'
        finally:
            self._stop.set(); timer.join(timeout=.2); self._set_elapsed(started)
            self.busy = False; self.namespace['step5_research_busy'] = False
            self._enable()

    # PLOTTING LOGIC: 固定案例小表画到PNG，完整manifest保留本地文件。
    def _show_cases(self):
        fig = Figure(figsize=(16, 8), facecolor='white'); ax = fig.subplots(); ax.axis('off')
        keys = [c for c in ['case_number', 'selection', 'cusip', 'side', 'day', 'ISSUER', 'SECTOR'] if c in self.case_manifest]
        shown = self.case_manifest[keys].copy()
        if 'day' in shown: shown['day'] = pd.to_datetime(shown.day).dt.strftime('%Y-%m-%d')
        fig.suptitle('Fixed case navigation | begin with 2 random, 2 typical, 2 high impact', fontsize=14)
        if len(shown):
            table = ax.table(cellText=shown.fillna('').astype(str).values, colLabels=keys, loc='center', cellLoc='left')
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.55)
        fig.text(.07, .04, 'Cases guide local Step3/4 inspection; selected cases do not estimate population rates. No-quote cases remain valid coverage evidence.', fontsize=10)
        self._render(fig, 'fixed_trade_cases')

    # CACHEING LOGIC: 已有sidecar不重算；先核对event_cache_key，保存失败可重试保存。
    def _build(self):
        from quote_quality_movement import build_movement_features
        from quote_quality_incremental import save_movement_sidecar
        if self.sidecar is None:
            if self.event_cache is None:
                raise ValueError('Run this add-on in the original idle Step5 kernel with step5_event_cache; no raw-quote rebuild is automatic.')
            # CORE LOGIC: STEP 1 — 拒绝不属于原实验的事件缓存
            # Input: metadata.event_cache_key=old-run，live event_cache.cache_key=other-run。
            # Output: ValueError，尚未调用build_movement_features；key相等且非空才继续。
            # Trick: 即使可用的event表有相同schema，也不能用不同source替换原实验。
            expected = self.metadata.get('event_cache_key')
            if not expected or self.event_cache.get('cache_key') != expected:
                raise ValueError('Live event cache does not match the saved Step5 event_cache_key.')
            # CORE LOGIC: STEP 2 — 固定30分钟配置构建movement
            # Input: frame含Train/Validation/Test，event_cache key与原实验一致；t=10:30,target A两端D1中心60→62且quantity相同。
            # Output: sidecar仅Train/Validation，A的bid_move_guarded_mean_bps=+2；30min lookback/age、allow_exact=True、issuer扩展均固定。
            # Trick: 调用共享事件表，不重新聚合raw；精确同timestamp沿用原known-time口径。
            self.sidecar = build_movement_features(self.frame, self.event_cache,
                lookback_min=30, age_min=30, allow_exact=True, include_issuer=True, progress=self._report)
        # FILE IO LOGIC: 先保存sidecar再开放fit；保存后绘支持情况。
        if not self.sidecar_saved:
            path = save_movement_sidecar(self.frame, self.sidecar, self.metadata, folder=self.output)
            self.sidecar_saved = True
            self.status.value = escape(f'Movement features saved separately: {path}. Existing Step5 features were retained.')
        path = self._render(movement_coverage_figure(self.frame, self.sidecar, self._quote_dates()), 'movement_support')
        self.detail.value = escape(f'Coverage only; no new prediction gain has been measured. Saved {path}')

    # VALIDATION LOGIC: 必须未locked/test-exposed且已有完整保存的sidecar。
    def _fit(self, block):
        from quote_quality_incremental import run_incremental_validation
        from quote_quality_diagnostics import diagnostics, figure
        # CORE LOGIC: STEP 1 — 防止界面状态过期后绕过fit前置约束
        # Input: _locked()=True,sidecar已保存；或_locked()=False但sidecar_saved=False。
        # Output: 前者因实验已锁抛ValueError；后者因特征未完整保存抛ValueError；均不会fit。
        # Trick: 按钮disabled只是UI提示，真正执行时仍重查锁。
        if self._locked(): raise ValueError('The existing experiment is locked or test was exposed; no new validation fit.')
        if self.sidecar is None or not self.sidecar_saved: raise ValueError('Build and save movement features first, then review their support.')
        # CORE LOGIC: STEP 2 — 每个新block最多执行一次候选验证
        # Input: block=Direction；已有预测仅Base/Quote levels/Reliability/Age decay；sidecar已保存，原参数/provenance有效。
        # Output: run_incremental_validation只为Age decay + Direction拟合或恢复一次；new预测附到原表。
        # Trick: 原四模型不重训；已存在同名模型时跳过此调用，新模型与saved Age decay同row_id比较。
        name = f'Age decay + {block}'
        if name not in set(self.predictions.model):
            new, _, meta = run_incremental_validation(self.frame, self.sidecar, self.original_predictions,
                self._metadata_now(), block, progress=self._report, folder=self.output)
            self.predictions = pd.concat([self.predictions, new], ignore_index=True)
            # UI LOGIC: 更新选项/完成状态，显示同样本误差和尾部诊断。
            self.selected.options = self.predictions.model.drop_duplicates().tolist()
            self.status.value = escape(f'{"Restored" if meta.get("reused") else "Completed one fit"}: {name}. Saved {meta["snapshot_path"]}')
        self.selected.value = name
        # CORE LOGIC: STEP 3 — 新候选显式对照已保存Age decay
        # Input: name=Age decay + Direction；相同Validation row_ids=[7,8]，新模型绝对误差bps=[1,2]，Age decay=[2,4]，实验字段完整。
        # Output: n=2，逐笔差[-1,-2]，整体平均差-1.5bps；这里只计算验证诊断。
        # Trick: 参考始终是原Age decay；不会误用新候选自身或另一个试验的样本。
        result = diagnostics(self.frame, self.predictions, selected=name, reference='Age decay')
        # PLOTTING LOGIC: 导出同样本诊断完整PNG及CSV；不访问test。
        path = self._render(figure(result), 'incremental_validation_' + block.lower())
        self._export_diagnostics(result, path)
        self.detail.value = 'Review all-row, daily, state and tail effects. The original four models were not retrained; no locked test was run.'


# UI LOGIC: 显示控件入口；不自动build、fit或访问locked test。
def show_research(namespace=None, folder='outputs/quote_quality_step5', output_folder='outputs/quote_quality_incremental'):
    """Display controls only. No automatic feature building, fitting or test access."""
    from IPython.display import display
    session = ResearchSession(namespace, folder, output_folder)
    display(session.dashboard)
    return session
