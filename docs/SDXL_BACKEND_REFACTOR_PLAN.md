# SDXL Backend Refactoring Plan

**File**: `backends/sdxl_backend/backend.py` (805 lines)  
**Priority**: HIGH  
**Target**: Split into 4 focused modules  
**Goal**: Reduce main file to ~200 lines

---

## Current Issues

1. **Mixed Concerns**: Model discovery, scheduler config, generation, utilities all in one file
2. **Size**: 805 lines - far exceeds guidelines
3. **Duplicate Code**: Model checkpoint mapping appears 3 times
4. **Maintainability**: Hard to modify scheduler list without scrolling through 290 lines

---

## Proposed Architecture

### Main File: `backend.py` (~200 lines)
**Responsibility**: SDXLBackend class and API interface
- Class definition and initialization
- Public API methods (init_backend, generate, unload_backend)
- Imports from extracted modules

### New Module 1: `model_loader.py` (~180 lines)
**Responsibility**: Model discovery and loading
- Model checkpoint mapping (single source of truth)
- Local file search logic
- Pipeline initialization
- HuggingFace fallback

**Functions**:
```python
get_model_checkpoint_map() -> dict
find_local_checkpoint(model_id, models_dir) -> Optional[str]
load_pipeline(checkpoint_path, config) -> Pipeline
is_model_downloaded(model_id) -> bool
get_model_file_details(model_id) -> tuple
```

### New Module 2: `schedulers.py` (~320 lines)
**Responsibility**: Scheduler and sampler configuration
- Scheduler metadata definitions
- Scheduler factory functions
- Sampler metadata

**Functions**:
```python
get_available_schedulers() -> list[dict]
get_available_samplers() -> list[dict]
create_scheduler(name, config) -> Scheduler
```

### New Module 3: `image_generator.py` (~120 lines)
**Responsibility**: Image generation logic
- Text2Image generation
- Img2Img generation
- Result encoding
- VRAM tracking

**Functions**:
```python
generate_text2img(pipeline, params) -> dict
generate_img2img(pipeline, params) -> dict
encode_results(images) -> list[str]
track_vram_usage() -> dict
```

---

## File Structure (After)

```
backends/sdxl_backend/
├── backend.py              (200 lines - Main class)
├── model_loader.py         (180 lines - Discovery & loading)
├── schedulers.py           (320 lines - Config & metadata)
└── image_generator.py      (120 lines - Generation logic)
```

**Total**: ~820 lines (distributed)  
**Max file**: 320 lines ✅  
**Reduction in main**: 605 lines removed (75%)

---

## Benefits

1. **Single Source of Truth**: Model checkpoint map in one place
2. **Easy to Extend**: Add new schedulers in dedicated file
3. **Testable**: Can test each module independently
4. **Clear Separation**: Each file has one responsibility
5. **Maintainable**: Easy to find and modify code

---

## Migration Strategy

### Phase 1: Extract Schedulers (~30 min)
1. Create `schedulers.py`
2. Move scheduler/sampler metadata
3. Move `set_scheduler()`, `get_available_schedulers()`, `get_available_samplers()`
4. Update imports in main file
5. Test: Verify schedulers list correctly

### Phase 2: Extract Model Loader (~30 min)
1. Create `model_loader.py`
2. Move model checkpoint map
3. Move file discovery logic
4. Move `is_model_downloaded()`, `get_model_file_details()`
5. Refactor `initialize()` to use new module
6. Test: Verify model loading works

### Phase 3: Extract Image Generator (~20 min)
1. Create `image_generator.py`
2. Move text2img/img2img logic
3. Move result encoding
4. Refactor `generate_image()` to use new module
5. Test: Verify generation works

### Phase 4: Cleanup Main File (~10 min)
1. Remove extracted code
2. Add imports
3. Update class methods to use modules
4. Final testing

---

## Testing Checklist

After each phase:
- [ ] Backend initializes correctly
- [ ] Model discovery works
- [ ] Schedulers list returns correctly
- [ ] Image generation (text2img) works
- [ ] Image generation (img2img) works
- [ ] VRAM tracking functional
- [ ] Unload cleans up properly

---

## Code Quality Improvements

While refactoring, also:
1. **DRY**: Remove duplicate model checkpoint maps
2. **Type Hints**: Add type annotations where missing
3. **Constants**: Extract magic numbers (e.g., search roots)
4. **Error Handling**: Consistent error messages
5. **Logging**: Consistent log format

---

**Status**: Ready to begin  
**Next Action**: Create schedulers.py (Phase 1)  
**Estimated Time**: 1.5 hours total
