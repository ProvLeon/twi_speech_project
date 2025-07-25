#!/usr/bin/env python3
"""
Environment Variable Checker for Twi Speech Training Pipeline

This utility helps debug environment variable loading issues.
Run this script to verify that your .env file is being loaded correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_env_file():
    """Check if .env file exists and display its contents (masked)."""
    print("🔍 Environment File Checker")
    print("=" * 50)

    # Find .env file
    current_dir = Path(__file__).parent
    env_file = current_dir / ".env"

    print(f"Looking for .env file at: {env_file}")

    if not env_file.exists():
        print("❌ .env file not found!")
        print(f"Expected location: {env_file}")
        print("\nTo create .env file:")
        print("1. Copy the template from .env.example (if exists)")
        print("2. Or create manually with required variables")
        return False

    print("✅ .env file found!")
    print(f"File size: {env_file.stat().st_size} bytes")

    # Read and display .env contents (masked)
    print("\n📄 .env file contents (values masked):")
    print("-" * 40)

    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Mask the value for security
                    masked_value = '*' * min(len(value), 8) if value else '(empty)'
                    print(f"  {key}={masked_value}")
                else:
                    print(f"  {line}")
            elif line.startswith('#'):
                print(f"  {line}")
            elif line:
                print(f"  {line}")

    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

    return True

def load_and_check_env():
    """Load environment variables and check if they're accessible."""
    print("\n🔄 Loading Environment Variables")
    print("=" * 50)

    # Load from .env file
    current_dir = Path(__file__).parent
    env_file = current_dir / ".env"

    if env_file.exists():
        print(f"Loading from: {env_file}")
        load_dotenv(dotenv_path=env_file)
        print("✅ .env file loaded using python-dotenv")
    else:
        print("⚠️  No .env file found, using system environment variables only")

    # Check required variables
    required_vars = [
        'MONGODB_URI',
        'MONGO_DB_NAME',
        'CLOUDFLARE_ACCOUNT_ID',
        'CLOUDFLARE_ACCESS_KEY_ID',
        'CLOUDFLARE_SECRET_ACCESS_KEY',
        'R2_BUCKET_NAME',
        'WANDB_API_KEY'
    ]

    print("\n📋 Environment Variable Status:")
    print("-" * 40)

    found_vars = 0
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'URI' in var or 'SECRET' in var:
                display_value = f"{'*' * 8}...{value[-4:]}" if len(value) > 4 else '*' * len(value)
            else:
                display_value = value[:20] + '...' if len(value) > 20 else value

            print(f"  ✅ {var} = {display_value}")
            found_vars += 1
        else:
            print(f"  ❌ {var} = (not set)")
            missing_vars.append(var)

    print(f"\n📊 Summary: {found_vars}/{len(required_vars)} variables found")

    if missing_vars:
        print(f"\n⚠️  Missing variables: {', '.join(missing_vars)}")
        return False
    else:
        print("\n🎉 All required environment variables are set!")
        return True

def test_database_connection():
    """Test if we can connect to MongoDB with loaded credentials."""
    print("\n🔌 Testing Database Connection")
    print("=" * 50)

    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        print("❌ MONGODB_URI not found, skipping connection test")
        return False

    try:
        from pymongo import MongoClient
        print("Attempting MongoDB connection...")
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        client.close()
        print("✅ MongoDB connection successful!")
        return True
    except ImportError:
        print("⚠️  pymongo not installed, skipping MongoDB test")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def test_s3_connection():
    """Test if we can connect to Cloudflare R2 with loaded credentials."""
    print("\n☁️  Testing Cloudflare R2 Connection")
    print("=" * 50)

    account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
    access_key = os.getenv('CLOUDFLARE_ACCESS_KEY_ID')
    secret_key = os.getenv('CLOUDFLARE_SECRET_ACCESS_KEY')

    if not all([account_id, access_key, secret_key]):
        print("❌ Missing Cloudflare R2 credentials, skipping connection test")
        return False

    try:
        import boto3
        from botocore.exceptions import ClientError

        print("Attempting Cloudflare R2 connection...")
        s3_client = boto3.client(
            service_name='s3',
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto"
        )

        # Test with list_buckets (lightweight operation)
        s3_client.list_buckets()
        print("✅ Cloudflare R2 connection successful!")
        return True
    except ImportError:
        print("⚠️  boto3 not installed, skipping R2 test")
        return True
    except ClientError as e:
        print(f"❌ Cloudflare R2 connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Cloudflare R2 connection failed: {e}")
        return False

def test_wandb_connection():
    """Test Weights & Biases configuration."""
    print("\n📊 Testing Weights & Biases Configuration")
    print("=" * 50)

    api_key = os.getenv('WANDB_API_KEY')
    if not api_key or not api_key.strip():
        print("⚠️  WANDB_API_KEY not found or empty")
        print("This is optional - training will work without W&B logging")
        return True

    try:
        import wandb
        print("Testing W&B login...")
        wandb.login(key=api_key, verify=True)
        print("✅ Weights & Biases login successful!")
        return True
    except ImportError:
        print("⚠️  wandb not installed, skipping W&B test")
        return True
    except Exception as e:
        print(f"⚠️  W&B login failed: {e}")
        print("Training will continue without W&B logging")
        return True

def main():
    """Main function to run all checks."""
    print("🚀 Twi Speech Training Pipeline - Environment Checker")
    print("=" * 60)

    all_good = True

    # Check .env file
    env_file_ok = check_env_file()
    all_good &= env_file_ok

    # Load and check environment variables
    env_vars_ok = load_and_check_env()
    all_good &= env_vars_ok

    # Test connections (optional)
    if env_vars_ok:
        print("\n🔧 Testing Connections (Optional)")
        print("=" * 50)

        test_database_connection()
        test_s3_connection()
        test_wandb_connection()

    # Final summary
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Environment Check PASSED!")
        print("Your environment variables are properly configured.")
        print("\nNext steps:")
        print("1. Run: python -m src.pipeline --mode validate")
        print("2. If validation passes, run: python -m src.pipeline --mode train --use-hf")
    else:
        print("❌ Environment Check FAILED!")
        print("\nPlease fix the issues above before training.")
        print("\nCommon fixes:")
        print("1. Create/edit the .env file with your credentials")
        print("2. Ensure the .env file is in the src/ directory")
        print("3. Check that variable names match exactly")
        print("4. Remove any extra spaces or quotes around values")

    print("=" * 60)
    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main())
