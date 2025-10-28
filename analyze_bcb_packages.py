#!/usr/bin/env python3
"""Analyze BigCodeBench package distribution and category overlap."""

from datasets import load_dataset
import json
import ast

def analyze_bigcodebench():
    """Analyze package distribution across BigCodeBench categories."""
    
    print("Loading BigCodeBench dataset...")
    ds_full = load_dataset('Joschka/bigcodebench', name='bcb-full', split='train')
    
    # Initialize counters
    total_samples = len(ds_full)
    numpy_samples = []
    matplotlib_samples = []
    
    # Category mappings based on README documentation
    category_libs = {
        'Computation': ['math', 'numpy', 'scipy', 'pandas', 'sklearn', 'statistics', 
                       'sympy', 'tensorflow', 'keras', 'statsmodels'],
        'Visualization': ['matplotlib', 'seaborn', 'PIL', 'folium', 'plotly', 'cv2',
                         'wordcloud', 'mpl_toolkits', 'turtle', 'skimage'],
        'Cryptography': ['hashlib', 'cryptography', 'base64', 'rsa', 'Crypto', 'hmac',
                        'secrets', 'blake3'],
        'Network': ['requests', 'urllib', 'flask', 'django', 'socket', 'http', 
                   'smtplib', 'flask_mail', 'flask_restful', 'flask_login', 'flask_wtf',
                   'mechanize', 'ipaddress', 'ssl', 'sendgrid', 'python_http_client'],
        'System': ['os', 'sys', 'subprocess', 'pathlib', 'shutil', 'glob', 'psutil',
                  'platform', 'ctypes', 'threading', 'multiprocessing', 'logging',
                  'configparser', 'pickle', 'io', 'tarfile', 'zipfile', 'gzip'],
        'Time': ['datetime', 'time', 'pytz', 'dateutil', 'calendar', 'holidays'],
        'General': ['collections', 'itertools', 'functools', 'operator', 'bisect',
                   'heapq', 'queue', 'array', 'struct', 'json', 're', 'regex',
                   'string', 'random', 'typing', 'enum', 'inspect', 'types']
    }
    
    # Reverse mapping: lib -> categories
    lib_to_categories = {}
    for category, libs in category_libs.items():
        for lib in libs:
            if lib not in lib_to_categories:
                lib_to_categories[lib] = []
            lib_to_categories[lib].append(category)
    
    # Analyze each sample
    sample_categories = []
    
    for i, sample in enumerate(ds_full):
        task_id = sample['task_id']
        libs_str = sample['libs']
        
        # Parse the libs field
        try:
            libs = ast.literal_eval(libs_str) if isinstance(libs_str, str) else libs_str
        except:
            libs = libs_str if isinstance(libs_str, list) else []
        
        # Check for numpy and matplotlib
        has_numpy = 'numpy' in libs
        has_matplotlib = 'matplotlib' in libs
        
        if has_numpy:
            numpy_samples.append(task_id)
        if has_matplotlib:
            matplotlib_samples.append(task_id)
        
        # Determine categories for this sample
        categories = set()
        for lib in libs:
            if lib in lib_to_categories:
                categories.update(lib_to_categories[lib])
        
        sample_categories.append({
            'task_id': task_id,
            'libs': libs,
            'categories': list(categories),
            'has_numpy': has_numpy,
            'has_matplotlib': has_matplotlib
        })
    
    # Analyze numpy samples
    numpy_only_computation = 0
    numpy_with_other_categories = 0
    numpy_category_breakdown = {}
    
    for sample in sample_categories:
        if sample['has_numpy']:
            categories = sample['categories']
            if categories == ['Computation']:
                numpy_only_computation += 1
            else:
                numpy_with_other_categories += 1
                for cat in categories:
                    numpy_category_breakdown[cat] = numpy_category_breakdown.get(cat, 0) + 1
    
    # Analyze matplotlib samples
    matplotlib_only_visualization = 0
    matplotlib_with_other_categories = 0
    matplotlib_category_breakdown = {}
    
    for sample in sample_categories:
        if sample['has_matplotlib']:
            categories = sample['categories']
            if categories == ['Visualization']:
                matplotlib_only_visualization += 1
            else:
                matplotlib_with_other_categories += 1
                for cat in categories:
                    matplotlib_category_breakdown[cat] = matplotlib_category_breakdown.get(cat, 0) + 1
    
    # Count samples per category
    category_counts = {cat: 0 for cat in category_libs.keys()}
    for sample in sample_categories:
        for cat in sample['categories']:
            category_counts[cat] += 1
    
    # Generate report
    print("\n" + "="*60)
    print("BIGCODEBENCH PACKAGE ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total samples: {total_samples}")
    
    print(f"\n📁 Category Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat:15} {count:4} samples")
    
    print(f"\n🔢 NumPy Analysis:")
    print(f"   Total samples with numpy: {len(numpy_samples)}")
    print(f"   Samples with ONLY Computation category: {numpy_only_computation}")
    print(f"   Samples with multiple categories: {numpy_with_other_categories}")
    if numpy_category_breakdown:
        print(f"   Category breakdown for numpy samples:")
        for cat, count in sorted(numpy_category_breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"      {cat:15} {count:4} samples")
    
    print(f"\n📈 Matplotlib Analysis:")
    print(f"   Total samples with matplotlib: {len(matplotlib_samples)}")
    print(f"   Samples with ONLY Visualization category: {matplotlib_only_visualization}")
    print(f"   Samples with multiple categories: {matplotlib_with_other_categories}")
    if matplotlib_category_breakdown:
        print(f"   Category breakdown for matplotlib samples:")
        for cat, count in sorted(matplotlib_category_breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"      {cat:15} {count:4} samples")
    
    print(f"\n🎯 Key Findings:")
    numpy_percentage = (numpy_only_computation / len(numpy_samples) * 100) if numpy_samples else 0
    print(f"   - {numpy_percentage:.1f}% of numpy samples are ONLY in Computation category")
    
    matplotlib_percentage = (matplotlib_only_visualization / len(matplotlib_samples) * 100) if matplotlib_samples else 0
    print(f"   - {matplotlib_percentage:.1f}% of matplotlib samples are ONLY in Visualization category")
    
    print(f"\n💡 Conclusion:")
    if numpy_percentage < 100:
        print(f"   ❌ NumPy samples are NOT exclusively in Computation category")
        print(f"      ({numpy_with_other_categories} samples have multiple categories)")
    else:
        print(f"   ✅ All NumPy samples are exclusively in Computation category")
    
    if matplotlib_percentage < 100:
        print(f"   ❌ Matplotlib samples are NOT exclusively in Visualization category")
        print(f"      ({matplotlib_with_other_categories} samples have multiple categories)")
    else:
        print(f"   ✅ All Matplotlib samples are exclusively in Visualization category")
    
    print("\n" + "="*60)
    
    # Additional analysis: Show some examples of cross-category samples
    print("\n📝 Examples of Cross-Category Samples:")
    
    print("\n   NumPy samples with multiple categories (first 5):")
    count = 0
    for sample in sample_categories:
        if sample['has_numpy'] and len(sample['categories']) > 1:
            print(f"      {sample['task_id']}: {', '.join(sample['categories'])}")
            count += 1
            if count >= 5:
                break
    
    print("\n   Matplotlib samples with multiple categories (first 5):")
    count = 0
    for sample in sample_categories:
        if sample['has_matplotlib'] and len(sample['categories']) > 1:
            print(f"      {sample['task_id']}: {', '.join(sample['categories'])}")
            count += 1
            if count >= 5:
                break
    
    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_bigcodebench()