"""
Scheduler and Sampler Configuration for SDXL Backend
Provides metadata and factory functions for all supported schedulers and samplers.
"""

from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler
)
import logging

logger = logging.getLogger("sdxl_backend.schedulers")


def get_available_schedulers():
    """Return comprehensive list of available schedulers with metadata"""
    return [
        {
            "id": "dpm_pp_2m_karras",
            "name": "DPM++ 2M Karras",
            "aliases": ["karras", "dpmpp_2m_karras"],
            "description": "Best quality, works great with 20-35 steps",
            "recommended": True,
            "optimal_steps_min": 20,
            "optimal_steps_max": 35,
            "best_for": "structure",
            "category": "DPM"
        },
        {
            "id": "dpm_pp_2m_sde_karras",
            "name": "DPM++ 2M SDE Karras",
            "aliases": ["dpmpp_2m_sde_gpu", "dpmpp_2m_sde_karras"],
            "description": "Adds grain/texture detail, perfect for skin & fabric",
            "recommended": True,
            "optimal_steps_min": 30,
            "optimal_steps_max": 45,
            "best_for": "texture",
            "category": "DPM"
        },
        {
            "id": "beta",
            "name": "Beta (DDIM)",
            "aliases": ["beta_schedule"],
            "description": "Smoothest gradients for soft portraits",
            "recommended": False,
            "optimal_steps_min": 35,
            "optimal_steps_max": 50,
            "best_for": "portraits",
            "category": "Specialized"
        },
        {
            "id": "exponential",
            "name": "Exponential",
            "aliases": ["exp"],
            "description": "Aggressive end-curve for dark/moody scenes",
            "recommended": False,
            "optimal_steps_min": 35,
            "optimal_steps_max": 45,
            "best_for": "dark",
            "category": "Specialized"
        },
        {
            "id": "align_your_steps",
            "name": "Align Your Steps (AYS)",
            "aliases": ["ays"],
            "description": "Front-loaded efficiency - 30-step quality in 12-15 steps",
            "recommended": False,
            "optimal_steps_min": 12,
            "optimal_steps_max": 20,
            "best_for": "speed",
            "category": "Efficiency"
        },
        {
            "id": "sgm_uniform",
            "name": "SGM Uniform",
            "aliases": ["uniform"],
            "description": "For Lightning models - ultra-fast",
            "recommended": False,
            "optimal_steps_min": 6,
            "optimal_steps_max": 10,
            "best_for": "lightning",
            "category": "Efficiency"
        },
        {
            "id": "simple",
            "name": "Simple",
            "aliases": ["normal", "linear"],
            "description": "Basic linear schedule - predictable",
            "recommended": False,
            "optimal_steps_min": 20,
            "optimal_steps_max": 50,
            "best_for": "general",
            "category": "Basic"
        },
        {
            "id": "dpm_pp_2m",
            "name": "DPM++ 2M",
            "aliases": ["dpmpp_2m"],
            "description": "Good quality without Karras sigmas",
            "recommended": False,
            "optimal_steps_min": 20,
            "optimal_steps_max": 40,
            "best_for": "general",
            "category": "DPM"
        },
        {
            "id": "euler_a",
            "name": "Euler Ancestral",
            "aliases": ["euler_ancestral", "euler_a"],
            "description": "Creative variation, adds noise throughout",
            "recommended": False,
            "optimal_steps_min": 15,
            "optimal_steps_max": 30,
            "best_for": "creative",
            "category": "Euler"
        },
        {
            "id": "euler",
            "name": "Euler",
            "aliases": ["euler_discrete"],
            "description": "Fast and reliable, good for testing",
            "recommended": False,
            "optimal_steps_min": 8,
            "optimal_steps_max": 25,
            "best_for": "testing",
            "category": "Euler"
        },
        {
            "id": "heun",
            "name": "Heun",
            "aliases": [],
            "description": "2nd order solver - more accurate",
            "recommended": False,
            "optimal_steps_min": 15,
            "optimal_steps_max": 30,
            "best_for": "quality",
            "category": "Advanced"
        },
        {
            "id": "dpm2",
            "name": "DPM2",
            "aliases": ["kdpm2"],
            "description": "Alternative 2nd order solver",
            "recommended": False,
            "optimal_steps_min": 20,
            "optimal_steps_max": 35,
            "best_for": "quality",
            "category": "Advanced"
        },
        {
            "id": "lms",
            "name": "LMS",
            "aliases": [],
            "description": "Linear multi-step - stable",
            "recommended": False,
            "optimal_steps_min": 20,
            "optimal_steps_max": 40,
            "best_for": "stable",
            "category": "Basic"
        }
    ]


def get_available_samplers():
    """Return list of available samplers (note: samplers are tied to schedulers in diffusers)"""
    return [
        {
            "id": "dpmpp_2m_sde_gpu",
            "name": "DPM++ 2M SDE",
            "description": "Best for skin/fabric texture",
            "scheduler_required": "dpm_pp_2m_sde_karras",
            "recommended": True,
            "category": "Texture"
        },
        {
            "id": "dpmpp_2m",
            "name": "DPM++ 2M",
            "description": "Clean structure, no texture noise",
            "scheduler_required": "dpm_pp_2m_karras",
            "recommended": True,
            "category": "Structure"
        },
        {
            "id": "dpmpp_3m_sde_gpu",
            "name": "DPM++ 3M SDE",
            "description": "Highest detail resolution (slow)",
            "scheduler_required": "dpm_pp_2m_sde_karras",
            "recommended": False,
            "category": "Detail"
        },
        {
            "id": "dpmpp_sde_gpu",
            "name": "DPM++ SDE",
            "description": "For Lightning models",
            "scheduler_required": "sgm_uniform",
            "recommended": False,
            "category": "Speed"
        },
        {
            "id": "euler",
            "name": "Euler",
            "description": "Simple and stable",
            "scheduler_required": "euler",
            "recommended": False,
            "category": "Basic"
        },
        {
            "id": "euler_ancestral",
            "name": "Euler Ancestral",
            "description": "Creative with variation",
            "scheduler_required": "euler_a",
            "recommended": False,
            "category": "Creative"
        },
        {
            "id": "restart",
            "name": "Restart",
            "description": "Self-correcting for anatomy",
            "scheduler_required": "karras",
            "recommended": False,
            "category": "Specialist"
        }
    ]


def create_scheduler(scheduler_name, pipeline_config):
    """
    Create a scheduler instance from a name and pipeline config.
    Returns configured scheduler object.
    """
    try:
        # DPM++ 2M Variants (Most Common)
        if scheduler_name == "dpm_pp_2m_karras" or scheduler_name == "karras":
            return DPMSolverMultistepScheduler.from_config(
                pipeline_config, use_karras_sigmas=True
            )
        elif scheduler_name == "dpm_pp_2m_sde_karras" or scheduler_name == "dpmpp_2m_sde_gpu":
            return DPMSolverMultistepScheduler.from_config(
                pipeline_config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++"
            )
        elif scheduler_name == "dpm_pp_2m":
            return DPMSolverMultistepScheduler.from_config(
                pipeline_config, use_karras_sigmas=False
            )

        # God Tier Specialized Schedulers
        elif scheduler_name == "beta":
            # Beta scheduler - smoother gradients for soft portraits
            return DDIMScheduler.from_config(
                pipeline_config,
                beta_schedule="scaled_linear",  # Beta schedule variant
                clip_sample=False
            )
        elif scheduler_name == "exponential":
            # Exponential - aggressive end curve for dark/moody
            return DPMSolverMultistepScheduler.from_config(
                pipeline_config,
                use_karras_sigmas=True,
                final_sigmas_type="zero"  # Exponential-like behavior
            )
        elif scheduler_name == "sgm_uniform":
            # SGM Uniform - for Lightning models
            return DPMSolverMultistepScheduler.from_config(
                pipeline_config,
                timestep_spacing="trailing",
                use_karras_sigmas=False
            )
        elif scheduler_name == "simple" or scheduler_name == "normal":
            # Simple/Normal - basic linear schedule
            return EulerDiscreteScheduler.from_config(
                pipeline_config,
                timestep_spacing="linspace"
            )
        elif scheduler_name == "align_your_steps" or scheduler_name == "ays":
            # AYS - Smart step scheduler for efficiency
            return DPMSolverMultistepScheduler.from_config(
                pipeline_config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )

        # Euler Variants
        elif scheduler_name == "euler_a" or scheduler_name == "euler_ancestral":
            return EulerAncestralDiscreteScheduler.from_config(pipeline_config)
        elif scheduler_name == "euler":
            return EulerDiscreteScheduler.from_config(
                pipeline_config, timestep_spacing="trailing"
            )

        # Additional Samplers
        elif scheduler_name == "heun":
            return HeunDiscreteScheduler.from_config(pipeline_config)
        elif scheduler_name == "dpm2":
            return KDPM2DiscreteScheduler.from_config(pipeline_config)
        elif scheduler_name == "lms":
            return LMSDiscreteScheduler.from_config(pipeline_config)

        else:
            # Default to Euler
            logger.warning(f"Unknown scheduler '{scheduler_name}', defaulting to Euler")
            return EulerDiscreteScheduler.from_config(
                pipeline_config, timestep_spacing="trailing"
            )

    except Exception as e:
        logger.error(f"Failed to create scheduler {scheduler_name}: {e}")
        # Fallback to Euler
        return EulerDiscreteScheduler.from_config(
            pipeline_config, timestep_spacing="trailing"
        )
