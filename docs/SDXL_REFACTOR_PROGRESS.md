# SDXL Backend Refactoring Progress

**Date**: 2026-01-03  
**Status**: IN PROGRESS (Phase 1 Complete)  
**File**: `backends/sdxl_backend/backend.py` (805 → target: ~200 lines)

---

## ✅ Phase 1 Complete: Schedulers Extracted

**File Created**: `backends/sdxl_backend/schedulers.py` (296 lines)

**Extracted Functions**:
- `get_available_schedulers()` - 13 scheduler definitions with metadata
- `get_available_samplers()` - 7 sampler definitions with metadata
- `create_scheduler(name, config)` - Factory function for scheduler instances

**Benefits**:
- Single location for all scheduler/sampler metadata
- Easy to add new schedulers
- Testable in isolation
- 296 lines removed from main file

---

## ⏳ Remaining Phases

### Phase 2: Model Loader (Next)
**File**: `model_loader.py` (est. ~180 lines)

**To Extract**:
- Lines 45-62: Model checkpoint mapping
- Lines 64-175: Model discovery logic (Strategy A & B)  
- Lines 676-805: Utility functions (`is_model_downloaded`, `get_model_file_details`)
-normalize() helper function

**Dependencies**: None (pure utility functions)

---

### Phase 3: Image Generator
**File**: `image_generator.py` (est. ~120 lines)

**To Extract**:
- Lines 533-601: `generate_image()` method
- Text2Img logic
- Img2Img logic
- Result encoding
- VRAM tracking

**Dependencies**: Needs model_loader

---

### Phase 4: Main File Integration
**File**: `backend.py` (target: ~200 lines)

**Actions**:
1. Import new modules
2. Update `SDXLBackend.initialize()` to use `model_loader`
3. Update `SDXLBackend.set_scheduler()` to use `schedulers.create_scheduler()`
4. Update `SDXLBackend.generate_image()` to use `image_generator`
5. Update module-level functions to use extracted modules
6. Remove all extracted code
7. Final testing

---

## File Size Tracking

| File | Before | Current | Target | Status |
|------|--------|---------|--------|--------|
| backend.py | 805 | 805 | 200 | ⏳ Pending integration |
| schedulers.py | - | 296 | <350 | ✅ Created |
| model_loader.py | - | - | ~180 | ⏳ Next |
| image_generator.py | - | - | ~120 | ⏳ Pending |

**Progress**: Phase 1/4 complete (25%)  
**Lines Extracted**: 296 (from main file scope)

---

## Next Steps

**Immediate** (Session continues):
1. Create `model_loader.py` (Phase 2)
2. Create `image_generator.py` (Phase 3)
3. Integrate into main file (Phase 4)
4. Test backend initialization
5. Test image generation
6. Commit changes

**Alternative** (Pause and test):
1. Stop here and test current state
2. Ensure no breaking changes
3. Continue in next session

---

## Integration Notes

The main `backend.py` file will need these changes in Phase 4:

```python
# New imports at top
from .schedulers import create_scheduler, get_available_schedulers, get_available_samplers
from .model_loader import find_local_checkpoint, load_pipeline, is_model_downloaded, get_model_file_details
from .image_generator import generate_text2img, generate_img2img

# In SDXLBackend class:
def set_scheduler(self, scheduler_name):
    if not self.pipeline: return
    self.pipeline.scheduler = create_scheduler(scheduler_name, self.pipeline.scheduler.config)

def get_available_schedulers(self):
    return get_available_schedulers()  # Module function

def get_available_samplers(self):
    return get_available_samplers()  # Module function
```

---

## Testing Strategy

After integration (Phase 4):
1. Import backend module (syntax check)
2. Initialize backend with a model
3. List available schedulers
4. Generate a test image (text2img)
5. Generate a test image (img2img)
6. Verify VRAM tracking
7. Test unload_backend()

---

**Status**: Ready for Phase 2  
**Estimated Time Remaining**: 45 minutes  
**Confidence**: High (clean extraction so far)
